#pragma once

// Mirrored from Intersection/config.py
constexpr int WIDTH  = 750;
constexpr int HEIGHT = 750;

constexpr float SCALE = 12.0f;
constexpr float FPS = 60.0f;
constexpr float DT_DEFAULT = 1.0f / 60.0f;

constexpr float CAR_LENGTH = 54.0f;         // int(4.5m * 12)
constexpr float CAR_WIDTH = 24.0f;          // int(2.0m * 12)
constexpr float WHEELBASE = CAR_LENGTH;

constexpr float LANE_WIDTH_PX = 42.0f;      // int(3.5m * 12)
constexpr float CORNER_RADIUS = 84.0f;      // int(7m * 12)

constexpr float MAX_ACC = 15.0f;
constexpr float MAX_STEERING_ANGLE = 0.6108652381980153f; // radians(35)
constexpr float PHYSICS_MAX_SPEED = 8.0f;  // px/frame
