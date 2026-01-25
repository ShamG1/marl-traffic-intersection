#pragma once

struct RenderColor {
    float r;
    float g;
    float b;
    float a;
};

namespace RenderColors {
    inline constexpr RenderColor Background{34/255.f, 139/255.f, 34/255.f, 1.0f};

    inline constexpr RenderColor RoadSurface{60/255.f, 60/255.f, 60/255.f, 1.0f};
    inline constexpr RenderColor Grass{34/255.f, 139/255.f, 34/255.f, 1.0f};

    inline constexpr RenderColor CenterLineYellow{1.0f, 0.8f, 0.0f, 1.0f};
    inline constexpr RenderColor MarkingWhite{0.94f, 0.94f, 0.94f, 1.0f};

    inline constexpr RenderColor RouteCyan{0.0f, 1.0f, 1.0f, 0.8f};
    inline constexpr RenderColor TargetRed{1.0f, 0.0f, 0.0f, 1.0f};

    inline constexpr RenderColor TrafficBodyGray{150/255.f, 150/255.f, 150/255.f, 1.0f};
    inline constexpr RenderColor TrafficHeadBlack{0.0f, 0.0f, 0.0f, 1.0f};

    inline constexpr RenderColor AgentHeadMarker{200/255.f, 200/255.f, 200/255.f, 1.0f};

    inline constexpr RenderColor LidarRayGreen{0.0f, 1.0f, 0.0f, 0.35f};
    inline constexpr RenderColor LidarHitRed{1.0f, 0.0f, 0.0f, 1.0f};

#ifndef _WIN32
    inline constexpr RenderColor LaneIdIn{0.0f, 0.0f, 200/255.f, 1.0f};
    inline constexpr RenderColor LaneIdOut{200/255.f, 0.0f, 0.0f, 1.0f};
#else
    inline constexpr unsigned int LaneIdInRGB = 0x0000C8;
    inline constexpr unsigned int LaneIdOutRGB = 0xC80000;
#endif

    inline constexpr unsigned int HudTextRGB = 0xFFFFFF;

    inline constexpr RenderColor RoadBoundary{0.0f, 0.0f, 0.0f, 1.0f};
}

