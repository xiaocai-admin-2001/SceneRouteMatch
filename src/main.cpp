#include <chrono>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "HttpServer.h"
#include "requests.h"
#include "json.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using hv::Json;

namespace {

class RotatingLogger {
public:
    void init(const std::string& path, std::uintmax_t max_size, int max_files) {
        std::lock_guard<std::mutex> lock(mu_);
        path_ = path;
        max_size_ = max_size;
        max_files_ = max_files;
        std::filesystem::create_directories(std::filesystem::path(path_).parent_path());
    }

    void info(const std::string& msg) { write("info", msg); }
    void warn(const std::string& msg) { write("warning", msg); }
    void error(const std::string& msg) { write("error", msg); }

private:
    void rotate_if_needed() {
        if (path_.empty() || !std::filesystem::exists(path_)) return;
        std::error_code ec;
        if (std::filesystem::file_size(path_, ec) < max_size_ || ec) return;

        for (int i = max_files_ - 1; i >= 1; --i) {
            std::filesystem::path src = path_ + "." + std::to_string(i);
            std::filesystem::path dst = path_ + "." + std::to_string(i + 1);
            if (std::filesystem::exists(src)) {
                if (i + 1 > max_files_) {
                    std::filesystem::remove(src, ec);
                } else {
                    std::filesystem::rename(src, dst, ec);
                }
            }
        }
        std::filesystem::rename(path_, path_ + ".1", ec);
    }

    std::string timestamp() const {
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
        localtime_r(&t, &tm);
        std::ostringstream os;
        os << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "."
           << std::setw(3) << std::setfill('0') << ms.count();
        return os.str();
    }

    void write(const std::string& level, const std::string& msg) {
        std::lock_guard<std::mutex> lock(mu_);
        if (path_.empty()) return;
        rotate_if_needed();
        std::ofstream out(path_, std::ios::app);
        out << "[" << timestamp() << "] [" << level << "] " << msg << "\n";
    }

    std::mutex mu_;
    std::string path_ = "logs/road_segment_test_service.log";
    std::uintmax_t max_size_ = 10 * 1024 * 1024;
    int max_files_ = 5;
};

RotatingLogger g_logger;

class ResponseArchive {
public:
    void init(const std::string& root, bool enabled) {
        root_ = root;
        enabled_ = enabled;
    }

    void save(const std::string& endpoint, const std::string& body) {
        if (!enabled_) return;
        try {
            auto now = std::chrono::system_clock::now();
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            std::tm tm{};
            localtime_r(&t, &tm);
            std::ostringstream day;
            day << std::put_time(&tm, "%Y%m%d");
            std::filesystem::path dir = std::filesystem::path(root_) / day.str();
            std::filesystem::create_directories(dir);

            auto epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()).count();
            auto sequence = sequence_.fetch_add(1, std::memory_order_relaxed);
            std::filesystem::path path = dir /
                (endpoint + "_" + std::to_string(epoch_ms) + "_" +
                 std::to_string(sequence) + ".json");
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out) {
                g_logger.error("failed to create response archive: " + path.string());
                return;
            }
            out << body << "\n";
        } catch (const std::exception& e) {
            g_logger.error(std::string("failed to archive response: ") + e.what());
        }
    }

private:
    bool enabled_ = false;
    std::string root_ = "logs/responses";
    std::atomic<unsigned long long> sequence_{0};
};

ResponseArchive g_response_archive;

struct AppConfig {
    int port = 19000;
    int threads = 4;
    std::string road_service_url = "http://127.0.0.1:8990/api/extract_road";
    std::string feature_service_url = "http://127.0.0.1:19001/match";
    std::string scene_root = "/opt/RoadSegmentTestService/scene";
    std::string response_log_dir = "/opt/RoadSegmentTestService/logs/responses";
    bool response_archive_enabled = false;
    double match_threshold = 0.30;
};

struct SegmentResult {
    Json polygons = Json::array();
    int width = 0;
    int height = 0;
    double runtime_ms = 0.0;
};

struct FeatureResult {
    std::string result = "0";
    std::string best_image;
    double best_score = 0.0;
    double runtime_ms = 0.0;
    int compared_count = 0;
    Json direction = nullptr;
    Json distances = Json::array();
};

struct SceneCacheEntry {
    std::filesystem::file_time_type mtime;
    SegmentResult segment;
};

std::mutex g_scene_cache_mu;
std::map<std::string, SceneCacheEntry> g_scene_cache;
std::mutex g_distance_map_mu;

long long now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string getenv_or(const char* name, const std::string& def) {
    const char* value = std::getenv(name);
    return value && *value ? std::string(value) : def;
}

int getenv_int(const char* name, int def) {
    const char* value = std::getenv(name);
    if (!value || !*value) return def;
    try { return std::stoi(value); } catch (...) { return def; }
}

double getenv_double(const char* name, double def) {
    const char* value = std::getenv(name);
    if (!value || !*value) return def;
    try { return std::stod(value); } catch (...) { return def; }
}

std::string json_error(int code, const std::string& msg) {
    Json j;
    j["status_code"] = code;
    j["data"] = nullptr;
    j["msg"] = msg;
    return j.dump(2);
}

int write_json(HttpResponse* resp, int http_code, const Json& payload) {
    resp->content_type = APPLICATION_JSON;
    resp->body = payload.dump(2);
    return http_code;
}

int write_error(HttpResponse* resp, int http_code, const std::string& msg) {
    resp->content_type = APPLICATION_JSON;
    resp->body = json_error(http_code, msg);
    g_logger.error("response error http=" + std::to_string(http_code) + " msg=" + msg);
    return http_code;
}

std::string make_default_id() {
    return "test_" + std::to_string(now_ms());
}

bool read_file(const std::string& path, std::string* out, std::string* err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        *err = "failed to open image file: " + path;
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    *out = ss.str();
    if (out->empty()) {
        *err = "empty image file: " + path;
        return false;
    }
    return true;
}

bool fetch_http_image(const std::string& url, std::string* out, std::string* err) {
    auto resp = requests::get(url.c_str());
    if (!resp) {
        *err = "failed to fetch image url";
        return false;
    }
    if (resp->status_code < 200 || resp->status_code >= 300) {
        *err = "failed to fetch image url, http status " + std::to_string(resp->status_code);
        return false;
    }
    *out = resp->body;
    if (out->empty()) {
        *err = "empty image from url";
        return false;
    }
    return true;
}

bool decode_image_size(const std::string& image_bytes, int* width, int* height, std::string* err) {
    std::vector<uchar> buf(image_bytes.begin(), image_bytes.end());
    cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (img.empty()) {
        *err = "invalid or unsupported image data";
        return false;
    }
    *width = img.cols;
    *height = img.rows;
    return true;
}

bool is_safe_component(const std::string& value) {
    return !value.empty() &&
           value.find('/') == std::string::npos &&
           value.find('\\') == std::string::npos &&
           value.find("..") == std::string::npos;
}

bool write_file_atomic(
    const std::filesystem::path& path,
    const std::string& content,
    std::string* err) {
    try {
        std::filesystem::create_directories(path.parent_path());
        std::filesystem::path temp =
            path.parent_path() / (path.filename().string() + ".tmp." + std::to_string(now_ms()));
        {
            std::ofstream out(temp, std::ios::binary | std::ios::trunc);
            if (!out) {
                *err = "failed to create temp file";
                return false;
            }
            out.write(content.data(), static_cast<std::streamsize>(content.size()));
            if (!out) {
                *err = "failed to write temp file";
                std::error_code ec;
                std::filesystem::remove(temp, ec);
                return false;
            }
        }
        std::error_code ec;
        std::filesystem::rename(temp, path, ec);
        if (ec) {
            std::filesystem::remove(path, ec);
            ec.clear();
            std::filesystem::rename(temp, path, ec);
        }
        if (ec) {
            *err = "failed to replace file: " + ec.message();
            std::filesystem::remove(temp, ec);
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        *err = e.what();
        return false;
    }
}

Json read_distance_map_locked(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) return Json::object();
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open distance_map.json");
    Json value = Json::parse(in);
    if (!value.is_object()) {
        throw std::runtime_error("distance_map.json must be an object");
    }
    return value;
}

Json load_image_metadata(
    const AppConfig& cfg,
    const std::string& camera_id,
    const std::string& image_name) {
    Json metadata = {
        {"result", nullptr},
        {"direction", nullptr},
        {"distances", Json::array()}
    };
    if (image_name.empty()) return metadata;
    std::lock_guard<std::mutex> lock(g_distance_map_mu);
    std::filesystem::path map_path =
        std::filesystem::path(cfg.scene_root) / camera_id / "distance_map.json";
    try {
        Json value = read_distance_map_locked(map_path);
        if (value.contains(image_name) && value[image_name].is_array()) {
            metadata["distances"] = value[image_name];
        } else if (value.contains(image_name) && value[image_name].is_object()) {
            const Json& item = value[image_name];
            if (item.contains("result") && item["result"].is_string()) {
                metadata["result"] = item["result"];
            }
            if (item.contains("direction") && item["direction"].is_string()) {
                metadata["direction"] = item["direction"];
            }
            if (item.contains("distances") && item["distances"].is_array()) {
                metadata["distances"] = item["distances"];
            }
        }
    } catch (const std::exception& e) {
        g_logger.warn("failed to read distance map path=" + map_path.string() + " err=" + e.what());
    }
    return metadata;
}

bool load_image_from_request(HttpRequest* req, const Json& data, std::string* image_bytes, std::string* err) {
    std::string data_format = data.value("data_format", "binary");
    if (data_format.empty()) data_format = "binary";

    if (data_format == "binary") {
        const auto& form = req->GetForm();
        auto it = form.find("image");
        if (it == form.end() || it->second.content.empty()) {
            *err = "highway_road_segment 测试接口需要 image";
            return false;
        }
        *image_bytes = it->second.content;
        return true;
    }

    if (!data.contains("image") || data["image"].is_null() || !data["image"].is_string()) {
        *err = "highway_road_segment 测试接口需要 image";
        return false;
    }

    std::string image_ref = data["image"].get<std::string>();
    if (data_format == "local") {
        return read_file(image_ref, image_bytes, err);
    }
    if (data_format == "http") {
        return fetch_http_image(image_ref, image_bytes, err);
    }

    *err = "unsupported data_format: " + data_format;
    return false;
}

Json call_road_service(const AppConfig& cfg, const std::string& id, const std::string& image_bytes, double* runtime_ms) {
    auto start = std::chrono::steady_clock::now();
    auto out_req = std::make_shared<HttpRequest>();
    out_req->method = HTTP_POST;
    out_req->url = cfg.road_service_url + "?pictures_num=1";
    out_req->timeout = 600;
    out_req->content_type = MULTIPART_FORM_DATA;
    out_req->headers["Content-Type"] = "multipart/form-data; boundary=" DEFAULT_MULTIPART_BOUNDARY;
    out_req->SetFormData("id_list", std::string("[\"") + id + "\"]");
    out_req->form["images"] = hv::FormData("", "image.jpg");
    out_req->form["images"].content = image_bytes;

    auto out_resp = requests::request(out_req);
    auto end = std::chrono::steady_clock::now();
    *runtime_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (!out_resp) {
        throw std::runtime_error("road segment service request failed");
    }
    if (out_resp->status_code < 200 || out_resp->status_code >= 300) {
        throw std::runtime_error("road segment service http status " + std::to_string(out_resp->status_code) + ": " + out_resp->body);
    }
    return Json::parse(out_resp->body);
}

FeatureResult call_feature_service(
    const AppConfig& cfg,
    const std::string& camera_id,
    const std::string& image_bytes) {
    auto out_req = std::make_shared<HttpRequest>();
    out_req->method = HTTP_POST;
    out_req->url = cfg.feature_service_url;
    out_req->timeout = 120;
    out_req->content_type = MULTIPART_FORM_DATA;
    out_req->headers["Content-Type"] = "multipart/form-data; boundary=" DEFAULT_MULTIPART_BOUNDARY;
    out_req->SetFormData("camera_id", camera_id);
    out_req->form["image"] = hv::FormData("", "image.jpg");
    out_req->form["image"].content = image_bytes;

    auto out_resp = requests::request(out_req);
    if (!out_resp) {
        throw std::runtime_error("feature service request failed");
    }
    if (out_resp->status_code < 200 || out_resp->status_code >= 300) {
        throw std::runtime_error(
            "feature service http status " + std::to_string(out_resp->status_code) +
            ": " + out_resp->body);
    }
    Json response = Json::parse(out_resp->body);
    if (response.value("status_code", 500) != 200 ||
        !response.contains("data") || response["data"].is_null()) {
        throw std::runtime_error("feature service error: " + response.dump());
    }
    const Json& data = response["data"];
    FeatureResult result;
    result.result = data.value("result", "0");
    result.best_image = data.value("best_image", "");
    result.best_score = data.value("best_score", 0.0);
    result.runtime_ms = data.value("runtime_ms", 0.0);
    result.compared_count = data.value("compared_count", 0);
    if (data.contains("direction")) {
        result.direction = data["direction"];
    }
    if (data.contains("distances") && data["distances"].is_array()) {
        result.distances = data["distances"];
    }
    return result;
}

SegmentResult segment_image(const AppConfig& cfg, const std::string& id, const std::string& image_bytes) {
    int decoded_width = 0;
    int decoded_height = 0;
    std::string err;
    if (!decode_image_size(image_bytes, &decoded_width, &decoded_height, &err)) {
        throw std::runtime_error(err);
    }

    double measured_runtime_ms = 0.0;
    Json road_resp = call_road_service(cfg, id, image_bytes, &measured_runtime_ms);
    if (road_resp.value("status_code", 500) != 200) {
        throw std::runtime_error("road segment service error: " + road_resp.dump());
    }
    if (!road_resp.contains("results") || !road_resp["results"].is_array() || road_resp["results"].empty()) {
        throw std::runtime_error("road segment service returned no results");
    }

    Json item = road_resp["results"][0];
    if (item.contains("msg") && !item["msg"].is_null()) {
        throw std::runtime_error("road segment failed: " + item["msg"].dump());
    }

    SegmentResult result;
    result.polygons = item.value("polygons", Json::array());
    result.width = item.value("imageWidth", decoded_width);
    result.height = item.value("imageHeight", decoded_height);
    result.runtime_ms = road_resp.value("runtime_ms", measured_runtime_ms);
    return result;
}

bool is_image_file(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp";
}

std::vector<std::filesystem::path> list_scene_images(const std::filesystem::path& camera_dir) {
    std::vector<std::filesystem::path> files;
    if (!std::filesystem::exists(camera_dir) || !std::filesystem::is_directory(camera_dir)) {
        return files;
    }
    for (const auto& entry : std::filesystem::directory_iterator(camera_dir)) {
        if (entry.is_regular_file() && is_image_file(entry.path())) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

cv::Mat polygons_to_mask(const Json& polygons, int width, int height) {
    cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);
    if (!polygons.is_array() || width <= 0 || height <= 0) return mask;

    for (const auto& polygon : polygons) {
        if (!polygon.contains("points") || !polygon["points"].is_array()) continue;
        std::vector<cv::Point> pts;
        for (const auto& point : polygon["points"]) {
            if (!point.is_array() || point.size() < 2) continue;
            int x = static_cast<int>(std::lround(point[0].get<double>()));
            int y = static_cast<int>(std::lround(point[1].get<double>()));
            x = std::max(0, std::min(width - 1, x));
            y = std::max(0, std::min(height - 1, y));
            pts.emplace_back(x, y);
        }
        if (pts.size() >= 3) {
            std::vector<std::vector<cv::Point>> contours{pts};
            cv::fillPoly(mask, contours, cv::Scalar(255));
        }
    }
    return mask;
}

double polygon_iou(const SegmentResult& a, const SegmentResult& b) {
    if (a.width <= 0 || a.height <= 0 || b.width <= 0 || b.height <= 0) return 0.0;
    cv::Mat a_mask = polygons_to_mask(a.polygons, a.width, a.height);
    cv::Mat b_mask = polygons_to_mask(b.polygons, b.width, b.height);
    if (b_mask.size() != a_mask.size()) {
        cv::resize(b_mask, b_mask, a_mask.size(), 0, 0, cv::INTER_NEAREST);
    }
    cv::Mat inter_mask;
    cv::Mat union_mask;
    cv::bitwise_and(a_mask, b_mask, inter_mask);
    cv::bitwise_or(a_mask, b_mask, union_mask);
    double inter = static_cast<double>(cv::countNonZero(inter_mask));
    double uni = static_cast<double>(cv::countNonZero(union_mask));
    return uni <= 0.0 ? 0.0 : inter / uni;
}

SegmentResult segment_scene_image_cached(const AppConfig& cfg, const std::filesystem::path& path) {
    std::string key = path.string();
    auto mtime = std::filesystem::last_write_time(path);
    {
        std::lock_guard<std::mutex> lock(g_scene_cache_mu);
        auto it = g_scene_cache.find(key);
        if (it != g_scene_cache.end() && it->second.mtime == mtime) {
            return it->second.segment;
        }
    }

    std::string bytes;
    std::string err;
    if (!read_file(key, &bytes, &err)) {
        throw std::runtime_error(err);
    }
    SegmentResult segment = segment_image(cfg, path.stem().string(), bytes);
    {
        std::lock_guard<std::mutex> lock(g_scene_cache_mu);
        g_scene_cache[key] = SceneCacheEntry{mtime, segment};
    }
    return segment;
}

std::string leading_match_label(const std::filesystem::path& path) {
    std::string name = path.filename().string();
    if (!name.empty() && (name[0] == '1' || name[0] == '2')) {
        return std::string(1, name[0]);
    }
    return "0";
}

std::string configured_match_label(const AppConfig& cfg, const std::filesystem::path& path) {
    std::lock_guard<std::mutex> lock(g_distance_map_mu);
    try {
        Json value = read_distance_map_locked(path.parent_path() / "distance_map.json");
        std::string name = path.filename().string();
        if (value.contains(name) && value[name].is_object()) {
            std::string result = value[name].value("result", "");
            if (result == "0" || result == "1" || result == "2") {
                return result;
            }
        }
    } catch (const std::exception& e) {
        g_logger.warn("failed to read configured result path=" + path.string() + " err=" + e.what());
    }
    return leading_match_label(path);
}

int handle_highway_road_segment(const AppConfig& cfg, HttpRequest* req, HttpResponse* resp) {
    auto request_start = std::chrono::steady_clock::now();
    try {
        if (req->ContentType() != MULTIPART_FORM_DATA) {
            return write_error(resp, 400, "request must be multipart/form-data");
        }

        std::string data_str = req->GetFormData("data");
        if (data_str.empty()) {
            return write_error(resp, 400, "missing multipart field: data");
        }

        Json data;
        try {
            data = Json::parse(data_str);
        } catch (const std::exception& e) {
            return write_error(resp, 400, std::string("invalid data json: ") + e.what());
        }

        std::string id = data.value("id", "");
        if (id.empty()) id = make_default_id();
        g_logger.info("request start path=/api/test/highway_road_segment id=" + id);

        std::string image_bytes;
        std::string err;
        if (!load_image_from_request(req, data, &image_bytes, &err)) {
            return write_error(resp, 400, err);
        }

        int decoded_width = 0;
        int decoded_height = 0;
        if (!decode_image_size(image_bytes, &decoded_width, &decoded_height, &err)) {
            return write_error(resp, 422, err);
        }

        double measured_runtime_ms = 0.0;
        Json road_resp = call_road_service(cfg, id, image_bytes, &measured_runtime_ms);
        if (road_resp.value("status_code", 500) != 200) {
            return write_error(resp, 500, "road segment service error: " + road_resp.dump());
        }
        if (!road_resp.contains("results") || !road_resp["results"].is_array() || road_resp["results"].empty()) {
            return write_error(resp, 500, "road segment service returned no results");
        }

        Json item = road_resp["results"][0];
        if (item.contains("msg") && !item["msg"].is_null()) {
            return write_error(resp, 500, "road segment failed: " + item["msg"].dump());
        }

        Json polygons = item.value("polygons", Json::array());
        int image_width = item.value("imageWidth", decoded_width);
        int image_height = item.value("imageHeight", decoded_height);
        double runtime_ms = road_resp.value("runtime_ms", measured_runtime_ms);

        Json payload;
        payload["status_code"] = 200;
        payload["msg"] = nullptr;
        payload["data"] = {
            {"runtime_ms", runtime_ms},
            {"id", id},
            {"polygon_count", polygons.is_array() ? static_cast<int>(polygons.size()) : 0},
            {"image_width", image_width},
            {"image_height", image_height},
            {"debug_mask_path", nullptr},
            {"polygons", polygons.is_array() ? polygons : Json::array()}
        };
        auto request_end = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(request_end - request_start).count();
        g_logger.info(
            "request done id=" + id +
            " image=" + std::to_string(image_width) + "x" + std::to_string(image_height) +
            " polygons=" + std::to_string(polygons.is_array() ? static_cast<int>(polygons.size()) : 0) +
            " backend_ms=" + std::to_string(runtime_ms) +
            " total_ms=" + std::to_string(total_ms));
        return write_json(resp, 200, payload);
    } catch (const std::exception& e) {
        return write_error(resp, 500, e.what());
    }
}

int handle_highway_scene_images(const AppConfig& cfg, HttpRequest* req, HttpResponse* resp) {
    try {
        std::string camera_id = req->GetParam("camera_id");
        std::filesystem::path root(cfg.scene_root);
        if (!std::filesystem::exists(root)) {
            return write_error(resp, 404, "scene root not found");
        }

        if (!camera_id.empty()) {
            if (!is_safe_component(camera_id)) {
                return write_error(resp, 400, "invalid camera_id");
            }
            std::filesystem::path camera_dir = root / camera_id;
            if (!std::filesystem::is_directory(camera_dir)) {
                return write_error(resp, 404, "camera_id not found");
            }
            auto files = list_scene_images(camera_dir);
            Json distance_map;
            {
                std::lock_guard<std::mutex> lock(g_distance_map_mu);
                distance_map = read_distance_map_locked(camera_dir / "distance_map.json");
            }
            Json images = Json::array();
            int configured_count = 0;
            for (const auto& path : files) {
                std::string name = path.filename().string();
                Json distances = Json::array();
                Json result = nullptr;
                Json direction = nullptr;
                if (distance_map.contains(name) && distance_map[name].is_array()) {
                    distances = distance_map[name];
                    configured_count++;
                } else if (distance_map.contains(name) && distance_map[name].is_object()) {
                    const Json& item = distance_map[name];
                    if (item.contains("result") && item["result"].is_string()) {
                        result = item["result"];
                    }
                    if (item.contains("direction") && item["direction"].is_string()) {
                        direction = item["direction"];
                    }
                    if (item.contains("distances") && item["distances"].is_array()) {
                        distances = item["distances"];
                    }
                    configured_count++;
                }
                images.push_back({
                    {"image", name},
                    {"result", result},
                    {"direction", direction},
                    {"distance_count", static_cast<int>(distances.size())},
                    {"distances", distances}
                });
            }
            Json payload;
            payload["status_code"] = 200;
            payload["msg"] = nullptr;
            payload["data"] = {
                {"camera_id", camera_id},
                {"image_count", static_cast<int>(files.size())},
                {"configured_count", configured_count},
                {"images", images}
            };
            return write_json(resp, 200, payload);
        }

        Json cameras = Json::array();
        int total_images = 0;
        int total_configured = 0;
        for (const auto& entry : std::filesystem::directory_iterator(root)) {
            if (!entry.is_directory()) continue;
            std::string id = entry.path().filename().string();
            auto files = list_scene_images(entry.path());
            Json distance_map;
            {
                std::lock_guard<std::mutex> lock(g_distance_map_mu);
                try {
                    distance_map = read_distance_map_locked(entry.path() / "distance_map.json");
                } catch (...) {
                    distance_map = Json::object();
                }
            }
            int configured_count = 0;
            for (const auto& path : files) {
                std::string name = path.filename().string();
                if (distance_map.contains(name) &&
                    (distance_map[name].is_array() || distance_map[name].is_object())) {
                    configured_count++;
                }
            }
            total_images += static_cast<int>(files.size());
            total_configured += configured_count;
            cameras.push_back({
                {"camera_id", id},
                {"image_count", static_cast<int>(files.size())},
                {"configured_count", configured_count}
            });
        }
        std::sort(cameras.begin(), cameras.end(), [](const Json& a, const Json& b) {
            return a.value("camera_id", "") < b.value("camera_id", "");
        });
        Json payload;
        payload["status_code"] = 200;
        payload["msg"] = nullptr;
        payload["data"] = {
            {"camera_count", static_cast<int>(cameras.size())},
            {"image_count", total_images},
            {"configured_count", total_configured},
            {"cameras", cameras}
        };
        return write_json(resp, 200, payload);
    } catch (const std::exception& e) {
        return write_error(resp, 500, e.what());
    }
}

int handle_highway_scene_image(const AppConfig& cfg, HttpRequest* req, HttpResponse* resp) {
    try {
        if (req->ContentType() != MULTIPART_FORM_DATA) {
            return write_error(resp, 400, "request must be multipart/form-data");
        }
        std::string data_str = req->GetFormData("data");
        if (data_str.empty()) {
            return write_error(resp, 400, "missing multipart field: data");
        }
        Json data;
        try {
            data = Json::parse(data_str);
        } catch (const std::exception& e) {
            return write_error(resp, 400, std::string("invalid data json: ") + e.what());
        }

        std::string camera_id = data.value("camera_id", "");
        std::string image_name = data.value("image", "");
        std::string configured_result = data.value("result", "");
        std::string direction = data.value("direction", "");
        if (!is_safe_component(camera_id)) {
            return write_error(resp, 400, "invalid camera_id");
        }
        if (!is_safe_component(image_name) || !is_image_file(image_name)) {
            return write_error(resp, 400, "invalid image name");
        }
        if (configured_result != "0" && configured_result != "1" && configured_result != "2") {
            return write_error(resp, 400, "result must be 0, 1 or 2");
        }
        if (direction != "向上上行、向下下行" && direction != "向上下行、向下上行") {
            return write_error(resp, 400, "direction must be 向上上行、向下下行 or 向上下行、向下上行");
        }
        if (!data.contains("distances") || !data["distances"].is_array()) {
            return write_error(resp, 400, "distances must be an array");
        }

        const auto& form = req->GetForm();
        auto image_it = form.find("image");
        if (image_it == form.end() || image_it->second.content.empty()) {
            return write_error(resp, 400, "missing multipart field: image");
        }
        const std::string& image_bytes = image_it->second.content;
        int width = 0;
        int height = 0;
        std::string err;
        if (!decode_image_size(image_bytes, &width, &height, &err)) {
            return write_error(resp, 422, err);
        }

        Json distances = Json::array();
        for (const auto& item : data["distances"]) {
            if (!item.is_object() ||
                !item.contains("y") || !item["y"].is_number() ||
                !item.contains("distance") || !item["distance"].is_number()) {
                return write_error(resp, 400, "each distance item requires numeric y and distance");
            }
            double y = item["y"].get<double>();
            double distance = item["distance"].get<double>();
            if (!std::isfinite(y) || !std::isfinite(distance) ||
                y < 0.0 || y > static_cast<double>(height - 1) || distance < 0.0) {
                return write_error(resp, 400, "invalid y or distance value");
            }
            distances.push_back({{"y", y}, {"distance", distance}});
        }
        std::sort(distances.begin(), distances.end(), [](const Json& a, const Json& b) {
            return a.value("y", 0.0) < b.value("y", 0.0);
        });

        std::filesystem::path camera_dir = std::filesystem::path(cfg.scene_root) / camera_id;
        std::filesystem::path image_path = camera_dir / image_name;
        if (!write_file_atomic(image_path, image_bytes, &err)) {
            return write_error(resp, 500, "failed to save image: " + err);
        }

        {
            std::lock_guard<std::mutex> lock(g_distance_map_mu);
            std::filesystem::path map_path = camera_dir / "distance_map.json";
            Json distance_map;
            try {
                distance_map = read_distance_map_locked(map_path);
            } catch (const std::exception& e) {
                return write_error(resp, 500, e.what());
            }
            distance_map[image_name] = {
                {"result", configured_result},
                {"direction", direction},
                {"distances", distances}
            };
            if (!write_file_atomic(map_path, distance_map.dump(2) + "\n", &err)) {
                return write_error(resp, 500, "failed to save distance map: " + err);
            }
        }
        {
            std::lock_guard<std::mutex> lock(g_scene_cache_mu);
            g_scene_cache.erase(image_path.string());
        }

        Json payload;
        payload["status_code"] = 200;
        payload["msg"] = nullptr;
        payload["data"] = {
            {"camera_id", camera_id},
            {"image", image_name},
            {"result", configured_result},
            {"direction", direction},
            {"distances", distances}
        };
        g_logger.info(
            "scene image saved camera_id=" + camera_id +
            " image=" + image_name +
            " result=" + configured_result +
            " direction=" + direction +
            " distances=" + std::to_string(distances.size()));
        return write_json(resp, 200, payload);
    } catch (const std::exception& e) {
        return write_error(resp, 500, e.what());
    }
}

int handle_delete_highway_scene_image(const AppConfig& cfg, HttpRequest* req, HttpResponse* resp) {
    try {
        std::string camera_id = req->GetParam("camera_id");
        std::string image_name = req->GetParam("image");
        if (!is_safe_component(camera_id)) {
            return write_error(resp, 400, "invalid camera_id");
        }
        if (!is_safe_component(image_name) || !is_image_file(image_name)) {
            return write_error(resp, 400, "invalid image name");
        }

        std::filesystem::path camera_dir = std::filesystem::path(cfg.scene_root) / camera_id;
        std::filesystem::path image_path = camera_dir / image_name;
        if (!std::filesystem::is_regular_file(image_path)) {
            return write_error(resp, 404, "scene image not found");
        }

        std::string err;
        {
            std::lock_guard<std::mutex> lock(g_distance_map_mu);
            std::filesystem::path map_path = camera_dir / "distance_map.json";
            Json distance_map;
            try {
                distance_map = read_distance_map_locked(map_path);
            } catch (const std::exception& e) {
                return write_error(resp, 500, e.what());
            }

            std::error_code remove_error;
            bool removed = std::filesystem::remove(image_path, remove_error);
            if (!removed || remove_error) {
                return write_error(resp, 500, "failed to delete image: " + remove_error.message());
            }

            distance_map.erase(image_name);
            if (!write_file_atomic(map_path, distance_map.dump(2) + "\n", &err)) {
                return write_error(resp, 500, "image deleted but failed to update distance map: " + err);
            }
        }
        {
            std::lock_guard<std::mutex> lock(g_scene_cache_mu);
            g_scene_cache.erase(image_path.string());
        }

        Json payload;
        payload["status_code"] = 200;
        payload["msg"] = nullptr;
        payload["data"] = {
            {"camera_id", camera_id},
            {"image", image_name},
            {"deleted", true}
        };
        g_logger.info("scene image deleted camera_id=" + camera_id + " image=" + image_name);
        return write_json(resp, 200, payload);
    } catch (const std::exception& e) {
        return write_error(resp, 500, e.what());
    }
}

int handle_highway_road_match(const AppConfig& cfg, HttpRequest* req, HttpResponse* resp) {
    auto request_start = std::chrono::steady_clock::now();
    try {
        if (req->ContentType() != MULTIPART_FORM_DATA) {
            return write_error(resp, 400, "request must be multipart/form-data");
        }

        std::string data_str = req->GetFormData("data");
        if (data_str.empty()) {
            return write_error(resp, 400, "missing multipart field: data");
        }

        Json data;
        try {
            data = Json::parse(data_str);
        } catch (const std::exception& e) {
            return write_error(resp, 400, std::string("invalid data json: ") + e.what());
        }

        std::string camera_id = data.value("camera_id", "");
        if (camera_id.empty()) camera_id = data.value("cameraId", "");
        if (camera_id.empty()) {
            return write_error(resp, 400, "missing camera_id");
        }
        if (camera_id.find('/') != std::string::npos || camera_id.find('\\') != std::string::npos ||
            camera_id.find("..") != std::string::npos) {
            return write_error(resp, 400, "invalid camera_id");
        }

        double threshold = data.value("threshold", cfg.match_threshold);
        threshold = std::max(0.0, std::min(1.0, threshold));
        g_logger.info("match request start camera_id=" + camera_id + " threshold=" + std::to_string(threshold));

        std::string image_bytes;
        std::string err;
        if (!load_image_from_request(req, data, &image_bytes, &err)) {
            return write_error(resp, 400, err);
        }

        FeatureResult feature = call_feature_service(cfg, camera_id, image_bytes);
        bool matched = !feature.best_image.empty() && feature.best_score >= threshold;
        std::string result = matched ? feature.result : "0";
        std::string final_best_image = matched ? feature.best_image : "";
        double total_score = matched ? feature.best_score : 0.0;
        Json direction = matched ? feature.direction : Json(nullptr);
        Json distances = matched ? feature.distances : Json::array();
        auto request_end = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(request_end - request_start).count();

        Json payload;
        payload["status_code"] = 200;
        payload["msg"] = nullptr;
        payload["data"] = {
            {"camera_id", camera_id},
            {"result", result},
            {"best_image", final_best_image.empty() ? Json(nullptr) : Json(final_best_image)},
            {"total_score", total_score},
            {"direction", direction},
            {"distances", distances}
        };
        g_logger.info(
            "match request done camera_id=" + camera_id +
            " result=" + result +
            " feature_result=" + feature.result +
            " matched=" + std::string(matched ? "true" : "false") +
            " best_image=" + final_best_image +
            " total_score=" + std::to_string(total_score) +
            " compared=" + std::to_string(feature.compared_count) +
            " total_ms=" + std::to_string(total_ms));
        return write_json(resp, 200, payload);
    } catch (const std::exception& e) {
        return write_error(resp, 500, e.what());
    }
}

}  // namespace

int main() {
    g_logger.init("logs/road_segment_test_service.log", 10 * 1024 * 1024, 5);
    AppConfig cfg;
    cfg.port = getenv_int("EVENT_TEST_PORT", cfg.port);
    cfg.threads = getenv_int("EVENT_TEST_THREADS", cfg.threads);
    cfg.road_service_url = getenv_or("ROAD_SEGMENT_URL", cfg.road_service_url);
    cfg.feature_service_url = getenv_or("FEATURE_SERVICE_URL", cfg.feature_service_url);
    cfg.scene_root = getenv_or("SCENE_ROOT", cfg.scene_root);
    cfg.response_log_dir = getenv_or("RESPONSE_LOG_DIR", cfg.response_log_dir);
    cfg.response_archive_enabled = getenv_int("RESPONSE_ARCHIVE_ENABLED", 0) != 0;
    cfg.match_threshold = getenv_double("MATCH_THRESHOLD", cfg.match_threshold);
    g_response_archive.init(cfg.response_log_dir, cfg.response_archive_enabled);
    g_logger.info("Logger initialized successfully");

    HttpService router;
    router.GET("/ping", [](HttpRequest*, HttpResponse* resp) {
        resp->body = "pong";
        return 200;
    });
    router.POST("/api/test/highway_road_match", [&cfg](HttpRequest* req, HttpResponse* resp) {
        int status = handle_highway_road_match(cfg, req, resp);
        g_response_archive.save("highway_road_match", resp->body);
        return status;
    });
    hv::HttpServer server(&router);
    server.setPort(cfg.port);
    server.setThreadNum(cfg.threads);

    std::cout << "road_segment_test_service listening on 0.0.0.0:" << cfg.port << std::endl;
    std::cout << "road segment backend: " << cfg.road_service_url << std::endl;
    g_logger.info("service start port=" + std::to_string(cfg.port) +
                  " backend=" + cfg.road_service_url +
                  " scene_root=" + cfg.scene_root +
                  " match_threshold=" + std::to_string(cfg.match_threshold));
    return server.run();
}

