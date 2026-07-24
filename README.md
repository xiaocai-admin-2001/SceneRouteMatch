# road_segment_test_service

C++ libhv/OpenCV adapter for `POST /api/test/highway_road_segment`.

It receives the V1 test-protocol multipart request, validates/decodes the image with OpenCV, calls the existing road segmentation backend:

`POST http://127.0.0.1:8990/api/extract_road?pictures_num=1`

and converts the backend response to the protocol shape:

```json
{
  "status_code": 200,
  "data": {
    "runtime_ms": 0,
    "id": "...",
    "polygon_count": 0,
    "image_width": 0,
    "image_height": 0,
    "debug_mask_path": null,
    "polygons": []
  },
  "msg": null
}
```

## Build

```bash
cd /home/cpk/road_segment_test_service
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

## Run

Default port is `19000` to avoid the existing Python service on `9000`.

```bash
EVENT_TEST_PORT=19000 ./build/road_segment_test_service
```

Environment variables:

- `EVENT_TEST_PORT`: service port, default `19000`
- `EVENT_TEST_THREADS`: libhv worker threads, default `4`
- `ROAD_SEGMENT_URL`: backend URL, default `http://127.0.0.1:8990/api/extract_road`

## Test

```bash
curl -X POST "http://127.0.0.1:19000/api/test/highway_road_segment" \
  -F 'data={"id":"road_seg_test_001","data_format":"binary","image":null}' \
  -F "image=@/path/to/image.jpg"
```
# -
