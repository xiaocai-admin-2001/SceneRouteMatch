RoadSegmentTestService

Endpoints:
  GET  /ping
  POST /api/test/highway_road_segment

Default:
  port: 19000
  road backend: http://127.0.0.1:8990/api/extract_road

Operations:
  ./startRoadSegmentTestService start
  ./startRoadSegmentTestService stop
  ./startRoadSegmentTestService restart
  ./startRoadSegmentTestService status
