#include <glm/glm.hpp>

enum class Pattern { Plume, Random, Rising, Falling, Circle }; // Emitter patterns
enum class ParticleType { Smoke, Spark, Firework, Rain, Snow }; // Particle types

// Individual particle struct
struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    float life;    // remaining life in seconds
    float maxLife; // Initial lifespan
    glm::vec4 colour;    
    float size;     // rendered pixel size
    ParticleType type;
};

// GPU buffer upload format
struct DrawParticle {
    glm::vec4 pos_size;  // xyz = position, w = size
    glm::vec4 colour;
};

// Firework struct
struct Firework {
    glm::vec3 pos, vel;
    float fuse; // Time before explosion
    glm::vec4 themeColour;
};

// Camera/mouse interaction struct
struct Camera {
    float yaw = -0.3f;
    float pitch = -0.2f;
    float dist = 8.0f;

    glm::vec3 target = glm::vec3(0,1,0);
    bool rotating = false, panning = false;
    double lastX = 0.0, lastY = 0.0;
};

// Gust of wind struct
struct Gust {
    glm::vec3 dir = glm::normalize(glm::vec3(1,0,0));
    glm::vec3 dirTarget = glm::vec3(1,0,0);
    float base = 0.2f;
    float current = 0.0f;
    float target = 0.0f;
    float timer = 0.0f;
} gust;

// Emitter struct that will spawn the particles
struct Emitter {
    glm::vec3 position = {0,0,0};
    glm::vec3 velocity = {0,5,0};
    float rate = 5000.0f; // particles per second
    float spread = glm::radians(15.0f); // in radians
    float lifeMean = 2.0f;
    float lifeJitter = 0.5f;
    float sizeMean = 6.0f;
    float sizeJitter = 2.0f;
    glm::vec4 colour = {1, 0.7f, 0.2f, 1};

    Pattern pattern = Pattern::Plume;
    ParticleType type = ParticleType::Smoke;

    float circleRadius = 1.5f;
    glm::vec2 areaExtents = {2,2};
    float speedScale = 1.0f;
};
