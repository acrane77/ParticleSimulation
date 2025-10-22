#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>
#include "particle_main.hpp"
using namespace std;

// Circular queue for storing live particles
class ParticleSystem { // Circular queue of particles
    private:
        size_t maxParticles;
        vector<Particle> particles;
        size_t head;
        size_t tail;
        size_t count;
    public:
        ParticleSystem() : maxParticles(50000), head(0), tail(0), count(0) { 
            particles.resize(maxParticles); // Reserve space for 50k particles
        }

        // Adds or overwrites to back of queue
        void spawnParticle(const Particle& particle) {
            if (count == maxParticles) {
                particles[head] = particle;
                head = (head + 1) % maxParticles; // Advance head
                tail = (tail + 1) % maxParticles; // overwrite oldest
            } 
            else {
                particles[head] = particle;
                head = (head + 1) % maxParticles; // Advance head
                count++;
            }
        }

        // Removes at back of queue
        void killOldestParticle() {
            if (count > 0) {
                tail = (tail + 1) % maxParticles; // Advance tail
                count--;
            }
        }

        // Iterates through live particles 
        template<typename T>
        void forEachAlive(T&& fn) {
            for (size_t i = 0, idx = tail; i < count; ++i, idx = (idx + 1) % maxParticles) {
                fn(particles[idx]);
            }
        }

        // Iterates through live particles
        template<typename T>
        void forEachAlive(T&& fn) const {
            for (size_t i = 0, idx = tail; i < count; ++i, idx = (idx + 1) % maxParticles) {
                fn(static_cast<const Particle&>(particles[idx]));
            }
        }

        size_t size() const { return count; } // Size getter
        bool isEmpty() const { return count == 0; } // Empty helper

        Particle& rear() { return particles[(head + maxParticles - 1) % maxParticles]; } // Rear getters
        const Particle& rear() const { return particles[(head + maxParticles - 1) % maxParticles]; }

        Particle& front() { return particles[tail]; } // Front getters
        const Particle& front() const { return particles[tail]; }
        
};
// random number generators
static mt19937 rng{ random_device{}() }; 
float g_spawnCarry = 0.0f;
inline float rand01() { return uniform_real_distribution<float>{ 0.0f, 1.0f }(rng); }
inline float randRange(float min, float max) { uniform_real_distribution<float> dist{ min, max }; return dist(rng); }

// Random 3D direction uniformly distributed on a sphere
inline glm::vec3 randomUnitVector() {
    const float z = randRange(-1.0f, 1.0f);
    const float t = 2.0f * glm::pi<float>() * rand01();
    const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    return glm::vec3(r * std::cos(t), z, r * std::sin(t));
}

// Random 2D point inside a unit disk
inline glm::vec2 randomInDisk() {
    float t = 2.0f * glm::pi<float>() * rand01();
    float r = std::sqrt(rand01());
    return { r*std::cos(t), r*std::sin(t) };
}

// Build orthonormal basis from given direction vector
inline glm::mat3 orthonormalBasisFromW(const glm::vec3& w) {
    glm::vec3 a = (std::abs(w.x) < 0.9f) ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
    glm::vec3 u = glm::normalize(glm::cross(a, w));
    glm:: vec3 v = glm::cross(w, u);
    return glm::mat3(u, v, w);
}

// Picks a random direction within a cone around the axis
inline glm::vec3 randomDirectionInCone(const glm::vec3& axis, float spread) {
    const float phi = 2.0f * glm::pi<float>() * rand01();
    const float cosMin = std::cos(spread);
    const float cosTheta = cosMin + (1.0f - cosMin) * rand01();
    const float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));

    const glm::mat3 basis = orthonormalBasisFromW(axis);
    const glm::vec3 u = basis[0];
    const glm::vec3 v = basis[1];
    const glm::vec3 w = basis[2];

    glm::vec3 direction = w * cosTheta + (u * std::cos(phi) + v * sin(phi)) * sinTheta;
    return glm::normalize(direction);
}

// Spawns new particles from an emitter
inline void spawnFromEmitter(ParticleSystem& ps, const Emitter& em, int toSpawn) {
    const float speed = glm::length(em.velocity * std::max(0.001f, em.speedScale));
    glm::vec3 axis = (speed > 0.0f) ? (em.velocity / speed) : glm::vec3(0, 1, 0);
    const float spread = glm::clamp(em.spread, 0.0f, glm::pi<float>());

    for (int i = 0; i < toSpawn; ++i) {
        Particle p{};
        p.type = em.type;
        p.colour = em.colour;

        float life = em.lifeMean + randRange(-em.lifeJitter, em.lifeJitter);
        p.maxLife = p.life = std::max(life, 0.05f);
        float size = em.sizeMean + randRange(-em.sizeJitter, em.sizeJitter);
        p.size = std::max(size, 1.0f);

        // Pattern determines the position/velocity/struct features
        switch (em.pattern) {
            case Pattern::Plume: {
                p.position = em.position;
                glm::vec3 dir = randomDirectionInCone(axis, spread);
                p.velocity = dir * speed;
                break;
            }
            // "Random" pattern
            case Pattern::Random: {
                glm::vec2 off = { randRange(-em.areaExtents.x, em.areaExtents.x), randRange(-em.areaExtents.y, em.areaExtents.y) };
                p.position = em.position + glm::vec3(off.x, 0.0f, off.y);
                glm::vec3 dir = randomUnitVector();
                p.velocity = dir * speed * 0.5f;
                break;
            }
            // "Rising" pattern
            case Pattern::Rising: {
                p.position = em.position + glm::vec3(randRange(-0.2f, 0.2f), 0.0f, randRange(-0.2f, 0.2f));
                glm::vec3 dir = glm::normalize(glm::vec3(randRange(-0.2f, 0.2f), 1.0f, randRange(-0.2f, 0.2f)));
                p.velocity = dir * speed * 0.8f;
                break;
            }
            // "Falling" pattern
            case Pattern::Falling: {
                p.position = em.position + glm::vec3(randRange(-em.areaExtents.x, em.areaExtents.x), randRange(0.5f, 1.5f), randRange(-em.areaExtents.y, em.areaExtents.y));
                glm::vec3 dir = glm::normalize(glm::vec3(randRange(-0.2f, 0.2f), -1.0f, randRange(-0.2f, 0.2f)));
                p.velocity = dir * speed * 1.2f;
                break;
            }
            // "Circle" pattern
            case Pattern::Circle: {
                float theta = randRange(0.0f, 2.0f * glm::pi<float>());
                glm::vec3 onCircle = em.position + glm::vec3(std::cos(theta), 0.0f, std::sin(theta)) * em.circleRadius;
                p.position = onCircle;
                glm::vec3 outward = glm::normalize(onCircle - em.position);
                glm::vec3 tangent = glm::normalize(glm::cross(glm::vec3(0,1,0), outward));
                p.velocity = glm::normalize(outward * 0.8f + tangent * 0.2f) * speed * 0.9f;
                break;
            }

        }
        // Adjust colours/size based on the type of particle
        if (p.type == ParticleType::Smoke) {
            p.colour = glm::vec4(0.8f, 0.8f, 0.8f, 0.9f);
        } 
        else if (p.type == ParticleType::Spark) {
            p.colour = glm::vec4(randRange(0.6f, 1.0f), randRange(0.3f, 1.0f), randRange(0.2f, 1.0f), 1.0f); // Glow-y red/orange/yellow 
            p.size = std::max(2.0f, p.size * 0.5f);
        }

        ps.spawnParticle(p);
    }
}
// Spawns a burst of particles for the firework explosion
inline void spawnFireworkBurst(ParticleSystem& ps, const glm::vec3& position, int count, float speedMin=0.6f, float speedMax=18.0f) {
    for (int i = 0; i < count; ++i) {
        Particle p{};
        p.type = ParticleType::Spark;
        p.position = position;
        glm::vec3 dir = randomUnitVector(); // Random outward direction
        float spd = randRange(speedMin, speedMax);
        p.velocity = dir * spd;
        p.maxLife = p.life = randRange(1.2f, 2.2f); // Random lifespan range
        p.size = randRange(2.0f, 4.0f);
        p.colour = glm::vec4(randRange(0.6f, 1.0f), randRange(0.3f, 1.0f), randRange(0.2f, 1.0f), 1.0f);
        ps.spawnParticle(p);
    }
}

// Wave function for turbulence
inline float bandWave(const glm::vec3& p, float t, float freq=0.35f, float speed=0.6f) {
    float phase = p.x*freq + p.z*freq + t*speed;
    return 0.5f * (sinf(phase) + 0.5f*sinf(.7f*phase + 1.7f));
}

// Main particle simulation step
inline void Step(ParticleSystem& ps, const std::vector<Emitter>& emitters, float dt, const glm::vec3& gravity, double& simTime) {
    // Emit new particles based on emitter rate
    for (const auto& em : emitters) {
        float want = em.rate * dt + g_spawnCarry;
        int toSpawn = (int)floor(want);
        g_spawnCarry = want - (float)toSpawn;
        if (toSpawn > 0) spawnFromEmitter(ps, em, toSpawn);
    }

    // Update each alive particle
    ps.forEachAlive([&](Particle& p) {
        glm::vec3 accel = gravity;

        const float smokeWeight = 0.12f;
        const float sparkWeight = 0.18f;

        // Compute wind direction & variation over time
        float angle = 0.35f * bandWave(p.position * .7f, (float)simTime, 0.7f, 1.6f);
        float cf = cosf(angle);
        float sf = sinf(angle);
        glm::vec3 localDir = glm::normalize(glm::vec3(cf*gust.dir.x + sf*gust.dir.z, 0.0f, -sf*gust.dir.x + cf*gust.dir.z));

        // Gust/turbulence
        float s = .5f + bandWave(p.position, (float)simTime, .6f, 1.8f) * .5f;
        glm::vec3 gustVec = localDir * (gust.base + gust.current *s);

        glm::vec3 turbulence = glm::vec3(0.6f * bandWave(p.position + glm::vec3(13,0,0), (float)simTime, 0.9f, 2.0f), 0.0f,
            0.6f * bandWave(p.position + glm::vec3(-7,0,0), (float)simTime, 0.9f, 1.6f));

        glm::vec3 windAccel = gustVec + turbulence;

        // Particle-specific physics and rendering
        if (p.type == ParticleType::Smoke) {
            accel *= smokeWeight; // Low grav
            accel += glm::vec3(0, 6.0f, 0); // buoyancy
            accel += windAccel * 2.0f; // strong winds
            p.velocity *= expf(-0.8f*dt); // drag
            p.size += 4.0f * dt; // Expansion over time (smoke blooms)
            p.life -= dt;

            float t = glm::clamp(1.0f - (p.life / p.maxLife), 0.0f, 1.0f);
            glm::vec3 cool = glm::vec3(0.8f);
            glm::vec3 warm = glm::vec3(0.6f);
            glm::vec3 rgb = glm::mix(cool, warm, t);
            p.colour = glm::vec4(rgb, glm::mix(0.9f, 0.0f, t)); // Fade out of existence
        }

        else if (p.type == ParticleType::Spark) {
            accel *= sparkWeight; // Moderate gravity, sparks dont rise as fast as smoke but they don't fall
            accel += glm::vec3(0, 2.0f, 0);
            accel += windAccel * 0.9f;
            float drag = expf(-2.0f * dt); // Drag
            p.velocity *= drag;
            p.life -= dt;

            float t = glm::clamp(1.0f - (p.life / p.maxLife), 0.0f, 1.0f);
            p.size = glm::mix(3.0f, 1.6f, t); // Shrink ast he sparks fade

            glm::vec3 hot = glm::vec3(1.0f, 0.55f, 0.15f); // Hot colour (new)
            glm::vec3 cool = glm::vec3(0.1f, 0.06f, 0.05f); // Cold colour (old)
            glm::vec3 rgb = glm::mix(hot, cool, t); // Mix depending on current life
            rgb.g *= (1.0f - 0.5f*t);
            p.colour = glm::vec4(rgb, 1.0f);

            float alpha = glm::mix(1.0f, 0.0f, t);
            p.colour.a = (p.life > 0.0f) ? glm::max(alpha, 0.08f) : 0.0f;

            // Brightness flicker
            float flicker = 1.0f + 0.1f * randRange(-1.0f, 1.0f);
            p.colour.r *= flicker;
            p.colour.g *= flicker;
            p.colour.b *= flicker;
        }

        else if (p.type == ParticleType::Rain) {
            accel *= 1.3f; // Faster falling
            accel += windAccel * .25f;
            p.velocity *= expf(-.2f *dt);
            p.life -= dt;

            if (p.position.y <= 0.0f) p.life = -1.0f; // Kill if it falls below plane
            p.size = glm::clamp(1.8f + 0.05f * glm::length(p.velocity), 1.6f, 4.0f);
        }
        else if (p.type == ParticleType::Snow) {
            accel *= .25f; // Slow falling
            accel += windAccel * 1.4f;

            // wobble/drift
            glm::vec3 wobble = glm::vec3(0.6f*bandWave(p.position + glm::vec3(3,0,0), (float)simTime, 1.1f, 1.8f), 0, 0.6f * bandWave(p.position + glm::vec3(-2,0,0), (float)simTime, 0.9f, 1.5f));
            accel += wobble;
            p.velocity *= expf(-0.6f * dt);
            p.life -= dt;

            // Fade and "melt" when touching ground
            if (p.position.y <= 0.0f) {
                p.velocity = glm::vec3(0);
                p.colour.a *= (1.0f - 4.0f * dt);
                if (p.colour.a < 0.05f) p.life = -1.0f;
            }
            p.size = glm::clamp(p.size + dt * randRange(-0.3f, 0.3f), 2.0f, 5.0f);
            }
        
        // Motion
        p.velocity += accel * dt;
        p.position += p.velocity * dt;
        
    });

    // Kill dead particles
    while (!ps.isEmpty() && ps.front().life <= 0.0f) {
        ps.killOldestParticle();
    }
}

// Build view matrix from camera pos
inline glm::mat4 cameraView(const Camera& camera) {
    float cp = std::cos(camera.pitch);
    float sp = std::sin(camera.pitch);

    float cy = std::cos(camera.yaw);
    float sy = std::sin(camera.yaw);

    glm::vec3 offset = glm::vec3(cp*sy, sp, cp*cy) * camera.dist;
    glm::vec3 eye = camera.target + offset;
    return glm::lookAt(eye, camera.target, glm::vec3(0,1,0));
}

// Convert screen mouse position to world ray direction
inline glm::vec3 screenRayDir(double mx, double my, int fbw, int fbh, const glm::mat4& invViewProj) {
    float x = float((2.0 * mx) / fbw - 1.0);
    float y = float(1.0 - (2.0 * my) / fbh);
    glm::vec4 pNear = invViewProj * glm::vec4(x, y, 0.0f, 1.0f);
    glm::vec4 pFar = invViewProj * glm::vec4(x, y, 1.0f, 1.0f);
    pNear /= pNear.w;
    pFar /= pFar.w;
    return glm::normalize(glm::vec3(pFar - pNear));
}

// Intersects ray with horizontal plane at Y
inline bool rayPlaneY(const glm::vec3& rayOrigin, const glm::vec3& rayDir, float yPlane, glm::vec3& hit) {
    if (std::abs(rayDir.y) < 1e-4f) return false; 
    float t = (yPlane - rayOrigin.y) / rayDir.y;
    if (t < 0.0f) return false;
    hit = rayOrigin + rayDir * t;
    return true;
}

// Compile OpenGL shader from source file
GLuint compileShader(GLenum type, const std::string& src) {
    GLuint shader = glCreateShader(type);
    const char* csrc = src.c_str();
    glShaderSource(shader, 1, &csrc, nullptr);
    glCompileShader(shader);
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        char buffer[512];
        glGetShaderInfoLog(shader, 512, nullptr, buffer);
        std::cerr << "Shader compile error: " << buffer << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

// Link vert and frag shaders to GPU
GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        char buffer[512];
        glGetProgramInfoLog(program, 512, nullptr, buffer);
        std::cerr << "Program link error: " << buffer << std::endl;
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

// Load text file contents (shader source) into a str
string loadFileAsString(const string& path) {
    ifstream file(path);
    if (!file.is_open()) {
        throw runtime_error("Failed to open file: " + path);
    }
    stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create OpenGL window
    GLFWwindow* window = glfwCreateWindow(800, 600, "Particles", nullptr, nullptr);
    if (!window) {glfwTerminate(); return -1; }

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return -1;

    glfwSwapInterval(1); // vsync
    glEnable(GL_PROGRAM_POINT_SIZE); // point size in shader
    glEnable(GL_BLEND); 
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.02f, 0.02f, 0.03f, 1.0f); // black background

    // Setup camera projection and view
    int fbw = 800, fbh = 600;
    glViewport(0, 0, fbw, fbh);
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), float(fbw)/float(fbh), 0.1f, 200.0f);
    Camera camera;

    static double gScrollY = 0.0;
    // Scrollwheel callback for zoom
    glfwSetScrollCallback(window, [](GLFWwindow*, double, double yoffset){ gScrollY += yoffset; });

    // VAO/VBO
    GLuint vao = 0, vbo = 0;
    const size_t MAX = 50000; // Maximum number of GPU draw particles

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, MAX * sizeof(DrawParticle), nullptr, GL_DYNAMIC_DRAW);
    
    // Attrib 0, pos/size
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(DrawParticle), (void*)offsetof(DrawParticle, pos_size));

    // Attrib 1, colour
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(DrawParticle), (void*)offsetof(DrawParticle, colour));
    glBindVertexArray(0);

    // Load shaders to string and compile
    string vertSrc = loadFileAsString("shaders/points.vert");
    string fragSrc = loadFileAsString("shaders/points.frag");

    GLuint vs = compileShader(GL_VERTEX_SHADER, vertSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    GLuint program = linkProgram(vs, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint loc = glGetUniformLocation(program, "uViewProj");

    // Simulation data & structures
    ParticleSystem ps;
    std::vector<Emitter> emitters;
    std::vector<Firework> fireworks;
    glm::vec3 gravity = {0, -9.81f, 0};
    std::vector<DrawParticle> drawBuffer(MAX);
    double simTime = 0.0;
    double fpsAccum = 0.0; int fpsFrames = 0; double fps = 0.0;

    // Global current type and pattern (default Smoke/Plume)
    ParticleType currentType = ParticleType::Smoke;
    Pattern currentPattern = Pattern::Plume;
    
    // Create emitter based on click position
    auto makeEmitterAt = [&](const glm::vec3& pos) {
        Emitter em;
        em.position = pos;
        em.pattern = currentPattern;
        em.type = currentType;
        
        // Create smoke emitter
        if (currentType == ParticleType::Smoke) {
            em.colour = glm::vec4(0.8f, 0.8f, 0.8f, 0.9f);
            em.velocity = glm::vec3(0,3.5f,0);
            em.rate = 2500.0f;
            em.spread = glm::radians(20.0f);
            em.lifeMean = 2.2f;
            em.lifeJitter = 0.6f;
            em.sizeMean = 8.0f;
            em.sizeJitter = 3.0f;
            if (em.pattern == Pattern::Falling) { em.velocity = glm::vec3(0, -2.5f, 0); }
            if (em.pattern == Pattern::Random) { em.areaExtents = {2.0f, 2.0f}; em.rate = 1800.0f; }
            if (em.pattern == Pattern::Circle) { em.circleRadius = 1.2f; em.rate = 2200.0f; }
            emitters.push_back(em);
        } 
        // Create spark emitter
        else if (currentType == ParticleType::Spark) {
            em.type = ParticleType::Spark;
            em.velocity = glm::vec3(0, 2.5f, 0);
            em.rate = 800.0f;
            if (em.pattern == Pattern::Falling) { em.velocity = glm::vec3(0, -2.5f, 0); }
            if (em.pattern == Pattern::Random) { em.areaExtents = {2.0f, 2.0f}; em.rate = 1800.0f; }
            if (em.pattern == Pattern::Circle) { em.circleRadius = 1.2f; em.rate = 2200.0f; }
            em.spread = glm::radians(18.0f);
            em.lifeMean = 1.6f;
            em.lifeJitter = 0.5f;
            em.sizeMean = 2.4f;
            em.sizeJitter = 0.6f;
            emitters.push_back(em);
        }
        // Create firework emitter w/ smoke trail
        else if (currentType == ParticleType::Firework) {
            Firework fw;
            fw.pos = pos;
            fw.vel = glm::vec3(randRange(-1.5f, 1.5f), randRange(12.0f, 18.0f), randRange(-1.5f, 1.5f));
            fw.fuse = randRange(1.4f, 2.2f);
            fw.themeColour = glm::vec4(randRange(0.6f, 1.0f), randRange(0.3f, 0.9f), randRange(.2f, .8f), 1.0f);
            fireworks.push_back(fw);

            Emitter trail;
            trail.position = pos;
            trail.type = ParticleType::Smoke;
            trail.pattern = Pattern::Rising;
            trail.velocity = glm::vec3(0, 2.5f, 0);
            trail.rate = 600.0f;
            trail.lifeMean = 0.6f;
            trail.lifeJitter = 0.2f;
            trail.sizeMean = 4.0f;
            trail.sizeJitter = 1.0f;
        }
        // Snow emitter
        else if (currentType == ParticleType::Snow) {
            Emitter em;
            em.type = ParticleType::Snow;
            em.pattern = Pattern::Falling;
            em.position = glm::vec3(pos.x, 12.0f, pos.z);
            em.areaExtents = {6.0f, 6.0f};
            em.velocity = glm::vec3(0, -1.8f, 0);
            em.rate = 1200.0f;
            em.spread = glm::radians(12.0f);
            em.lifeMean = 5; 
            em.lifeJitter = 1.5f;
            em.sizeMean = 3.5f;
            em.sizeJitter = 1;
            em.colour = glm::vec4(0.95f, 0.95f, 1, 0.85f);
            emitters.push_back(em);
        }
        // Rain emitter
        else if (currentType == ParticleType::Rain) {
            Emitter em;
            em.type = ParticleType::Rain;
            em.pattern = Pattern::Falling;
            em.position = glm::vec3(pos.x, 12.0f, pos.z);
            em.areaExtents = {6.0f, 6.0f};
            em.velocity = glm::vec3(0, -10.0f, 0);
            em.rate = 3000.0f;
            em.spread = glm::radians(4.0f);
            em.lifeMean = 2.5f;
            em.lifeJitter = 0.4f;
            em.sizeMean = 2.0f;
            em.sizeJitter = 0.5f;
            em.colour = glm::vec4(0.65f, 0.7f, .8f, .7f);
            emitters.push_back(em);
        }
    };

    // Setup time for steps
    double last = glfwGetTime(), acc = 0.0;
    const double dt = 1.0/120.0;

    // Track mb states
    bool prevLMB = false, prevRMB = false, prevMMB = false;

    // Main application loop
    while(!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        double frameDt = now - last;
        acc += now - last;
        last = now;
        
        // FPS Calculation
        fpsAccum += frameDt;
        fpsFrames += 1;
        if (fpsAccum >= 0.5) {
            fps = fpsFrames/fpsAccum;
            fpsAccum = 0.0; fpsFrames = 0;
        }

        // Read mouse position and buttons
        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        bool LMB = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        bool RMB = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        bool MMB = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
        bool shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS);

        // Read keyboard shortcuts for patterns
        if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) currentPattern = Pattern::Plume;
        if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) currentPattern = Pattern::Random;
        if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) currentPattern = Pattern::Rising;
        if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) currentPattern = Pattern::Falling;
        if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) currentPattern = Pattern::Circle;

        // Read keyboard shortcuts for types
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) currentType = ParticleType::Smoke;
        if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) currentType = ParticleType::Spark;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) currentType = ParticleType::Firework;
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) currentType = ParticleType::Rain;
        if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS) currentType = ParticleType::Snow;
        if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) { emitters.clear(); fireworks.clear();}

        // Camera rotation and panning
        if (RMB && !prevRMB) { camera.rotating = true; camera.lastX = mx; camera.lastY = my; }
        if (!RMB) camera.rotating = false;
        if ((MMB && !prevMMB) || (shift && RMB && !prevRMB)) { camera.panning = true; camera.lastX = mx; camera.lastY = my; }
        if (!(MMB || (shift && RMB))) camera.panning = false;

        // If camera is rotating (mouse movement)
        if (camera.rotating) {
            float dx = float(mx - camera.lastX);
            float dy = float(my - camera.lastY);
            camera.yaw -= dx * 0.005f;
            camera.pitch -= dy * 0.005f;
            camera.pitch = glm::clamp(camera.pitch, -1.2f, 1.2f);
            camera.lastX = mx;
            camera.lastY = my;
        }
        // If camera is panning (mmb or shift rmb)
        if (camera.panning) {
            float dx = float(mx - camera.lastX), dy = float(my - camera.lastY);
            glm::mat4 view = cameraView(camera);
            glm::vec3 right = glm::vec3(glm::row(glm::mat3(glm::transpose(view)), 0));
            glm::vec3 up = glm::vec3(0,1,0);
            float scale = 0.0025f * camera.dist;
            camera.target -= right * dx * scale;
            camera.target += up * dy * scale;
            camera.lastX = mx;
            camera.lastY = my;
        }
        // If camera is zooming (scroll)
        if (gScrollY != 0.0) {
            camera.dist *= std::pow(0.9f, gScrollY);
            camera.dist = glm::clamp(camera.dist, 2.0f, 80.0f);
            gScrollY = 0.0;
        }
        
        // Left clicking creates a new emitter once at location
        if (LMB && !prevLMB) {
            glm::mat4 view = cameraView(camera);
            glm::mat4 vp = proj * view;
            glm::mat4 invVP = glm::inverse(vp);

            float cp = std::cos(camera.pitch);
            float sp = std::sin(camera.pitch);
            float cy = std::cos(camera.yaw);
            float sy = std::sin(camera.yaw);
            glm::vec3 offset = glm::vec3(cp*sy, sp, cp*cy) * camera.dist;
            glm::vec3 eye = camera.target + offset;

            glm::vec3 dir = screenRayDir(mx, my, fbw, fbh, invVP);
            glm::vec3 hit;
            if (rayPlaneY(eye, dir, 0.0f, hit)) {
                makeEmitterAt(hit);
            }
        }
        prevLMB = LMB; prevRMB = RMB; prevMMB = MMB;

        // Update gust over time to change wind smoothly
        auto updateGust = [&](float dt) {
            gust.timer -= dt;
            if (gust.timer <= 0.0f) {
                gust.timer = randRange(2.0f, 5.0f); // new gust every 2 - 10 seconds
                gust.target = randRange(0.0f, 2.0f); // Strength of the new gust of wind

                // Random yaw rotation/direction
                float dYaw = randRange(-glm::radians(35.0f), glm::radians(35.0f));
                float c = cosf(dYaw);
                float s = sinf(dYaw);

                glm::vec3 d = gust.dir;
                gust.dirTarget = glm::normalize(glm::vec3(c*d.x + s*d.z, 0.0f, -s*d.x + c*d.z));

                gust.base = 0.45f; // Base intensity
            }
            // Blend gust 
            float k = 1.0f - expf(-dt/.8f);
            gust.current = glm::mix(gust.current, gust.target, k);

            // Blend wind direction
            float kd = 1.0f - expf(-dt/0.8f);
            glm::vec3 blended = glm::normalize(glm::mix(gust.dir, gust.dirTarget, kd));
            if (glm::all(glm::greaterThan(glm::abs(blended), glm::vec3(1e-4f)))) {
                gust.dir = blended;
            }
        };

        // Run updates while at timestep
        while(acc >= dt) { 
            updateGust((float)dt); // Update gusts
            simTime += dt; // Advance time
            Step(ps, emitters, (float)dt, gravity, simTime); // Advance particles
            acc -= dt; 

            // Update fireworks
            for (size_t i = 0; i < fireworks.size();) {
                Firework& fw = fireworks[i];

                glm::vec3 fwAccel = gravity * 0.8f + glm::vec3(0, 1.5f, 0);
                fw.vel += fwAccel * (float)dt;
                fw.pos += fw.vel * (float)dt;
                fw.fuse -= (float)dt;

                // Create smoke trail during flight
                if (((int)(simTime * 90)) % 2 == 0) {
                    Particle puff{};
                    puff.type = ParticleType::Smoke;
                    puff.position = fw.pos;
                    puff.velocity = glm::vec3(randRange(-0.3f, 0.3f), randRange(0.5f, 1.2f), randRange(-0.3f, 0.3f));
                    puff.maxLife = puff.life = randRange(.5f, .9f);
                    puff.size = randRange(3.0f, 6.0f);
                    puff.colour = glm::vec4(0.8f);
                    ps.spawnParticle(puff);
                }

                // Explode into sparks when fuse is 0
                if (fw.fuse <= 0.0f || fw.vel.y < 0.0f) {
                    spawnFireworkBurst(ps, fw.pos, 260, 6.0f, 16.0f);

                    fireworks[i] = fireworks.back();
                    fireworks.pop_back();
                    continue;
                }
                i++;
            }
        }

        // Sort live particles for rendering & counts
        size_t alive = ps.size(), j = 0; 
        size_t smokeCount = 0, sparkCount = 0, rainCount = 0, snowCount = 0;

        // Count each particle type
        ps.forEachAlive([&](const Particle& p) {
            switch (p.type) {
                case ParticleType::Smoke: smokeCount++; break;
                case ParticleType::Spark: sparkCount++; break;
                case ParticleType::Rain: rainCount++; break;
                case ParticleType::Snow: snowCount++; break;
                default: break;
            }
        });
        // Base indices for each particle in buffer
        size_t baseSmoke = 0;
        size_t baseSnow = baseSmoke + smokeCount;
        size_t baseRain = baseSnow  + snowCount;
        size_t baseSpark = baseRain  + rainCount;

        // Fill draw buffer by grouping particles
        size_t iSmoke = baseSmoke, iSnow = baseSnow, iRain = baseRain, iSpark = baseSpark;
        ps.forEachAlive([&](const Particle& p) {
            size_t idx;
            if (p.type == ParticleType::Smoke) idx = iSmoke++;
            else if (p.type == ParticleType::Snow) idx = iSnow++;
            else if (p.type == ParticleType::Rain) idx = iRain++;
            else idx = iSpark++;
            drawBuffer[idx].pos_size = glm::vec4(p.position, p.size);
            drawBuffer[idx].colour = p.colour;
        });
        
        // Send data to GPU
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, (baseSpark + sparkCount) * sizeof(DrawParticle), drawBuffer.data());

        // Draw frame
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   

        glm::mat4 view = cameraView(camera);
        glm::mat4 vp = proj * view;

        glUseProgram(program);
        glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(vp));
        glBindVertexArray(vao);

        // Transparent smoke/snow/rain first
        glDepthMask(GL_FALSE);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDrawArrays(GL_POINTS, (GLsizei)baseSmoke, (GLsizei)(smokeCount + snowCount + rainCount));

        // Spark particles
        glBlendFunc(GL_ONE, GL_ONE);
        glDrawArrays(GL_POINTS, (GLsizei)baseSpark, (GLsizei)sparkCount);

        // Restore default blending
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_TRUE);

        // Update window title with info
        {
            const char* patterns[] = { "Plume", "Random", "Rising", "Falling", "Circle" };
            const char* types[] = { "Smoke", "Spark", "Firework", "Rain", "Snow" };
            char title[128];
            snprintf(title, sizeof(title), "Particles | Pattern: %s | Type: %s | Alive: %zu | FPS: %.1f", patterns[(int)currentPattern], types[(int)currentType], alive, fps);
            glfwSetWindowTitle(window, title);
        }

        // Show frame and poll inputs
        glfwSwapBuffers(window);
        glfwPollEvents();

        // Window resize
        int newWinH, newWinW;
        glfwGetFramebufferSize(window, &newWinW, &newWinH);
        if (newWinW != fbw || newWinH != fbh) {
            fbw = newWinW; fbh = newWinH;
            glViewport(0, 0, fbw, fbh);
            proj = glm::perspective(glm::radians(60.0f), float(fbw)/float(fbh), 0.1f, 200.0f);
        }
    }

    // Cleanup after main loop ends
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}


