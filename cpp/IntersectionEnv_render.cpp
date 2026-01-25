#include "IntersectionEnv.h"
#include "Renderer.h"


void IntersectionEnv::render(bool show_lane_ids, bool show_lidar){
    if(!render_enabled){
        render_enabled=true;
    }
    if(!renderer){
        renderer=std::make_unique<Renderer>();
    }
    if(renderer && renderer->ok()){
        renderer->render(*this, show_lane_ids, show_lidar);
    }
}

bool IntersectionEnv::window_should_close() const {
    if(!renderer) return true;
    return renderer->window_should_close();
}

void IntersectionEnv::poll_events() const {
    if(!renderer) return;
    renderer->poll_events();
}

bool IntersectionEnv::key_pressed(int glfw_key) const {
    if(!renderer) return false;
    return renderer->key_pressed(glfw_key);
}
