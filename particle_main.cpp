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
using namespace std;

enum class Pattern { Plume, Random, Rising, Falling, Circle };
enum class ParticleType { Smoke, Spark, Firework, Rain, Snow };

struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    float life;    // in seconds
    float maxLife;
    glm::vec4 colour;    
    float size;     // in pixels
    ParticleType type;
};

struct DrawParticle {
    glm::vec4 pos_size;  // xyz = position, w = size
    glm::vec4 colour;
};

struct Firework {
    glm::vec3 pos, vel;
    float fuse;
    glm::vec4 themeColour;
};

struct Camera {
    float yaw = -0.3f;
    float pitch = -0.2f;
    float dist = 8.0f;

    glm::vec3 target = glm::vec3(0,1,0);
    bool rotating = false, panning = false;
    double lastX = 0.0, lastY = 0.0;
};

struct Gust {
    glm::vec3 dir = glm::normalize(glm::vec3(1,0,0));
    glm::vec3 dirTarget = glm::vec3(1,0,0);
    float base = 0.2f;
    float current = 0.0f;
    float target = 0.0f;
    float timer = 0.0f;
} gust;

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

        void killOldestParticle() {
            if (count > 0) {
                tail = (tail + 1) % maxParticles; // Advance tail
                count--;
            }
        }

        template<typename T>
        void forEachAlive(T&& fn) {
            for (size_t i = 0, idx = tail; i < count; ++i, idx = (idx + 1) % maxParticles) {
                fn(particles[idx]);
            }
        }

        template<typename T>
        void forEachAlive(T&& fn) const {
            for (size_t i = 0, idx = tail; i < count; ++i, idx = (idx + 1) % maxParticles) {
                fn(static_cast<const Particle&>(particles[idx]));
            }
        }

        size_t size() const { return count; }
        bool isEmpty() const { return count == 0; }

        Particle& rear() { return particles[(head + maxParticles - 1) % maxParticles]; }
        const Particle& rear() const { return particles[(head + maxParticles - 1) % maxParticles]; }

        Particle& front() { return particles[tail]; }
        const Particle& front() const { return particles[tail]; }
        
};

static mt19937 rng{ random_device{}() }; // random number generators
float g_spawnCarry = 0.0f;
inline float rand01() { return uniform_real_distribution<float>{ 0.0f, 1.0f }(rng); }
inline float randRange(float min, float max) { uniform_real_distribution<float> dist{ min, max }; return dist(rng); }

inline glm::vec3 randomUnitVector() {
    const float z = randRange(-1.0f, 1.0f);
    const float t = 2.0f * glm::pi<float>() * rand01();
    const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    return glm::vec3(r * std::cos(t), z, r * std::sin(t));
}

inline glm::vec2 randomInDisk() {
    float t = 2.0f * glm::pi<float>() * rand01();
    float r = std::sqrt(rand01());
    return { r*std::cos(t), r*std::sin(t) };
}

inline glm::mat3 orthonormalBasisFromW(const glm::vec3& w) {
    glm::vec3 a = (std::abs(w.x) < 0.9f) ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
    glm::vec3 u = glm::normalize(glm::cross(a, w));
    glm:: vec3 v = glm::cross(w, u);
    return glm::mat3(u, v, w);
}

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

        switch (em.pattern) {
            case Pattern::Plume: {
                p.position = em.position;
                glm::vec3 dir = randomDirectionInCone(axis, spread);
                p.velocity = dir * speed;
                break;
            }
            case Pattern::Random: {
                glm::vec2 off = { randRange(-em.areaExtents.x, em.areaExtents.x), randRange(-em.areaExtents.y, em.areaExtents.y) };
                p.position = em.position + glm::vec3(off.x, 0.0f, off.y);
                glm::vec3 dir = randomUnitVector();
                p.velocity = dir * speed * 0.5f;
                break;
            }
            case Pattern::Rising: {
                p.position = em.position + glm::vec3(randRange(-0.2f, 0.2f), 0.0f, randRange(-0.2f, 0.2f));
                glm::vec3 dir = glm::normalize(glm::vec3(randRange(-0.2f, 0.2f), 1.0f, randRange(-0.2f, 0.2f)));
                p.velocity = dir * speed * 0.8f;
                break;
            }
            case Pattern::Falling: {
                p.position = em.position + glm::vec3(randRange(-em.areaExtents.x, em.areaExtents.x), randRange(0.5f, 1.5f), randRange(-em.areaExtents.y, em.areaExtents.y));
                glm::vec3 dir = glm::normalize(glm::vec3(randRange(-0.2f, 0.2f), -1.0f, randRange(-0.2f, 0.2f)));
                p.velocity = dir * speed * 1.2f;
                break;
            }
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

        if (p.type == ParticleType::Smoke) {
            p.colour = glm::vec4(0.8f, 0.8f, 0.8f, 0.9f);
        } 
        else if (p.type == ParticleType::Spark) {
            p.colour = glm::vec4(randRange(0.6f, 1.0f), randRange(0.3f, 1.0f), randRange(0.2f, 1.0f), 1.0f);
            p.size = std::max(2.0f, p.size * 0.5f);
        }

        ps.spawnParticle(p);
    }
}

inline void spawnFireworkBurst(ParticleSystem& ps, const glm::vec3& position, int count, float speedMin=0.6f, float speedMax=18.0f) {
    for (int i = 0; i < count; ++i) {
        Particle p{};
        p.type = ParticleType::Spark;
        p.position = position;
        glm::vec3 dir = randomUnitVector();
        float spd = randRange(speedMin, speedMax);
        p.velocity = dir * spd;
        p.maxLife = p.life = randRange(1.2f, 2.2f);
        p.size = randRange(2.0f, 4.0f);
        p.colour = glm::vec4(randRange(0.6f, 1.0f), randRange(0.3f, 1.0f), randRange(0.2f, 1.0f), 1.0f);
        ps.spawnParticle(p);
    }
}

inline float bandWave(const glm::vec3& p, float t, float freq=0.35f, float speed=0.6f) {
    float phase = p.x*freq + p.z*freq + t*speed;
    return 0.5f * (sinf(phase) + 0.5f*sinf(.7f*phase + 1.7f));
}

inline void Step(ParticleSystem& ps, const std::vector<Emitter>& emitters, float dt, const glm::vec3& gravity, double& simTime) {
    for (const auto& em : emitters) {
        float want = em.rate * dt + g_spawnCarry;
        int toSpawn = (int)floor(want);
        g_spawnCarry = want - (float)toSpawn;
        if (toSpawn > 0) spawnFromEmitter(ps, em, toSpawn);
    }

    ps.forEachAlive([&](Particle& p) {
        glm::vec3 accel = gravity;

        const float smokeWeight = 0.12f;
        const float sparkWeight = 0.18f;

        float angle = 0.35f * bandWave(p.position * .7f, (float)simTime, 0.7f, 1.6f);
        float cf = cosf(angle);
        float sf = sinf(angle);
        glm::vec3 localDir = glm::normalize(glm::vec3(cf*gust.dir.x + sf*gust.dir.z, 0.0f, -sf*gust.dir.x + cf*gust.dir.z));

        float s = .5f + bandWave(p.position, (float)simTime, .6f, 1.8f) * .5f;
        glm::vec3 gustVec = localDir * (gust.base + gust.current *s);

        glm::vec3 turbulence = glm::vec3(0.6f * bandWave(p.position + glm::vec3(13,0,0), (float)simTime, 0.9f, 2.0f), 0.0f,
            0.6f * bandWave(p.position + glm::vec3(-7,0,0), (float)simTime, 0.9f, 1.6f));

        glm::vec3 windAccel = gustVec + turbulence;

        if (p.type == ParticleType::Smoke) {
            accel *= smokeWeight;
            accel += glm::vec3(0, 6.0f, 0);
            accel += windAccel * 2.0f;
            p.velocity *= expf(-0.8f*dt);
            p.size += 4.0f * dt;
            p.life -= dt;
            float t = glm::clamp(1.0f - (p.life / p.maxLife), 0.0f, 1.0f);
            glm::vec3 cool = glm::vec3(0.8f);
            glm::vec3 warm = glm::vec3(0.6f);
            glm::vec3 rgb = glm::mix(cool, warm, t);
            p.colour = glm::vec4(rgb, glm::mix(0.9f, 0.0f, t));
        }
        else if (p.type == ParticleType::Spark) {
            accel *= sparkWeight;
            accel += glm::vec3(0, 2.0f, 0);
            accel += windAccel * 0.9f;

            float drag = expf(-2.0f * dt);
            p.velocity *= drag;
            p.life -= dt;
            float t = glm::clamp(1.0f - (p.life / p.maxLife), 0.0f, 1.0f);
            p.size = glm::mix(3.0f, 1.6f, t);

            glm::vec3 hot = glm::vec3(1.0f, 0.55f, 0.15f);
            glm::vec3 cool = glm::vec3(0.1f, 0.06f, 0.05f);
            glm::vec3 rgb = glm::mix(hot, cool, t);
            rgb.g *= (1.0f - 0.5f*t);
            p.colour = glm::vec4(rgb, 1.0f);

            float alpha = glm::mix(1.0f, 0.0f, t);
            p.colour.a = (p.life > 0.0f) ? glm::max(alpha, 0.08f) : 0.0f;
            float flicker = 1.0f + 0.1f * randRange(-1.0f, 1.0f);
            p.colour.r *= flicker;
            p.colour.g *= flicker;
            p.colour.b *= flicker;
        }
        else if (p.type == ParticleType::Rain) {
            accel *= 1.3f;
            accel += windAccel * .25f;
            p.velocity *= expf(-.2f *dt);
            p.life -= dt;

            if (p.position.y <= 0.0f) p.life = -1.0f;
            p.size = glm::clamp(1.8f + 0.05f * glm::length(p.velocity), 1.6f, 4.0f);
        }
        else if (p.type == ParticleType::Snow) {
            accel *= .25f;
            accel += windAccel * 1.4f;

            glm::vec3 wobble = glm::vec3(0.6f*bandWave(p.position + glm::vec3(3,0,0), (float)simTime, 1.1f, 1.8f), 0,
                0.6f * bandWave(p.position + glm::vec3(-2,0,0), (float)simTime, 0.9f, 1.5f));
            accel += wobble;
            p.velocity *= expf(-0.6f * dt);
            p.life -= dt;

            if (p.position.y <= 0.0f) {
                p.velocity = glm::vec3(0);
                p.colour.a *= (1.0f - 4.0f * dt);
                if (p.colour.a < 0.05f) p.life = -1.0f;
            }
            p.size = glm::clamp(p.size + dt * randRange(-0.3f, 0.3f), 2.0f, 5.0f);
            }

        p.velocity += accel * dt;
        p.position += p.velocity * dt;
        
    });

    while (!ps.isEmpty() && ps.front().life <= 0.0f) {
        ps.killOldestParticle();
    }
}

inline glm::mat4 cameraView(const Camera& camera) {
    float cp = std::cos(camera.pitch);
    float sp = std::sin(camera.pitch);

    float cy = std::cos(camera.yaw);
    float sy = std::sin(camera.yaw);

    glm::vec3 offset = glm::vec3(cp*sy, sp, cp*cy) * camera.dist;
    glm::vec3 eye = camera.target + offset;
    return glm::lookAt(eye, camera.target, glm::vec3(0,1,0));
}

inline glm::vec3 screenRayDir(double mx, double my, int fbw, int fbh, const glm::mat4& invViewProj) {
    float x = float((2.0 * mx) / fbw - 1.0);
    float y = float(1.0 - (2.0 * my) / fbh);
    glm::vec4 pNear = invViewProj * glm::vec4(x, y, 0.0f, 1.0f);
    glm::vec4 pFar = invViewProj * glm::vec4(x, y, 1.0f, 1.0f);
    pNear /= pNear.w;
    pFar /= pFar.w;
    return glm::normalize(glm::vec3(pFar - pNear));
}

inline bool rayPlaneY(const glm::vec3& rayOrigin, const glm::vec3& rayDir, float yPlane, glm::vec3& hit) {
    if (std::abs(rayDir.y) < 1e-4f) return false; 
    float t = (yPlane - rayOrigin.y) / rayDir.y;
    if (t < 0.0f) return false;
    hit = rayOrigin + rayDir * t;
    return true;
}

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

string loadFIleAsString(const string& path) {
    ifstream file(path);
    if (!file.is_open()) {
        throw runtime_error("Failed to open file: " + path);
    }
    stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main() {
    // Initialization
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "Particles", nullptr, nullptr);
    if (!window) {glfwTerminate(); return -1; }

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return -1;

    glfwSwapInterval(1); 
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.02f, 0.02f, 0.03f, 1.0f);

    // Camera
    int fbw = 800, fbh = 600;
    glViewport(0, 0, fbw, fbh);
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), float(fbw)/float(fbh), 0.1f, 200.0f);
    Camera camera;

    static double gScrollY = 0.0;
    glfwSetScrollCallback(window, [](GLFWwindow*, double, double yoffset){ gScrollY += yoffset; });

    // VAO/VBO
    GLuint vao = 0, vbo = 0;
    const size_t MAX = 50000;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, MAX * sizeof(DrawParticle), nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(DrawParticle), (void*)offsetof(DrawParticle, pos_size));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(DrawParticle), (void*)offsetof(DrawParticle, colour));
    glBindVertexArray(0);

    // Shaders
    string vertSrc = loadFIleAsString("shaders/points.vert");
    string fragSrc = loadFIleAsString("shaders/points.frag");

    GLuint vs = compileShader(GL_VERTEX_SHADER, vertSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    GLuint program = linkProgram(vs, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint loc = glGetUniformLocation(program, "uViewProj");

    // Simulation
    ParticleSystem ps;
    std::vector<Emitter> emitters;
    std::vector<Firework> fireworks;
    glm::vec3 gravity = {0, -9.81f, 0};
    std::vector<DrawParticle> drawBuffer(MAX);
    double simTime = 0.0;
    double fpsAccum = 0.0; int fpsFrames = 0; double fps = 0.0;

    Pattern currentPattern = Pattern::Plume;
    ParticleType currentType = ParticleType::Smoke;

    auto makeEmitterAt = [&](const glm::vec3& pos) {
        Emitter em;
        em.position = pos;
        em.pattern = currentPattern;
        em.type = currentType;

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

    double last = glfwGetTime(), acc = 0.0;
    const double dt = 1.0/120.0;

    bool prevLMB = false, prevRMB = false, prevMMB = false;

    while(!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        double frameDt = now - last;
        acc += now - last;
        last = now;
        
        fpsAccum += frameDt;
        fpsFrames += 1;
        if (fpsAccum >= 0.5) {
            fps = fpsFrames/fpsAccum;
            fpsAccum = 0.0; fpsFrames = 0;
        }
        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        bool LMB = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        bool RMB = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        bool MMB = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
        bool shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS);

        if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) currentPattern = Pattern::Plume;
        if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) currentPattern = Pattern::Random;
        if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) currentPattern = Pattern::Rising;
        if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) currentPattern = Pattern::Falling;
        if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) currentPattern = Pattern::Circle;

        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) currentType = ParticleType::Smoke;
        if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) currentType = ParticleType::Spark;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) currentType = ParticleType::Firework;
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) currentType = ParticleType::Rain;
        if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS) currentType = ParticleType::Snow;
        if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) { emitters.clear(); fireworks.clear();}

        if (RMB && !prevRMB) { camera.rotating = true; camera.lastX = mx; camera.lastY = my; }
        if (!RMB) camera.rotating = false;
        if ((MMB && !prevMMB) || (shift && RMB && !prevRMB)) { camera.panning = true; camera.lastX = mx; camera.lastY = my; }
        if (!(MMB || (shift && RMB))) camera.panning = false;

        if (camera.rotating) {
            float dx = float(mx - camera.lastX);
            float dy = float(my - camera.lastY);
            camera.yaw -= dx * 0.005f;
            camera.pitch -= dy * 0.005f;
            camera.pitch = glm::clamp(camera.pitch, -1.2f, 1.2f);
            camera.lastX = mx;
            camera.lastY = my;
        }
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
        if (gScrollY != 0.0) {
            camera.dist *= std::pow(0.9f, gScrollY);
            camera.dist = glm::clamp(camera.dist, 2.0f, 80.0f);
            gScrollY = 0.0;
        }

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
        auto updateGust = [&](float dt) {
            gust.timer -= dt;
            if (gust.timer <= 0.0f) {
                gust.timer = randRange(2.0f, 5.0f); // new gust every 2 - 10 seconds
                gust.target = randRange(0.0f, 2.0f); // Strength of the new gust of wind

                float dYaw = randRange(-glm::radians(35.0f), glm::radians(35.0f));
                float c = cosf(dYaw);
                float s = sinf(dYaw);

                glm::vec3 d = gust.dir;
                gust.dirTarget = glm::normalize(glm::vec3(c*d.x + s*d.z, 0.0f, -s*d.x + c*d.z));

                gust.base = 0.45f;
            }
            float k = 1.0f - expf(-dt/.8f);
            gust.current = glm::mix(gust.current, gust.target, k);

            float kd = 1.0f - expf(-dt/0.8f);
            glm::vec3 blended = glm::normalize(glm::mix(gust.dir, gust.dirTarget, kd));
            if (glm::all(glm::greaterThan(glm::abs(blended), glm::vec3(1e-4f)))) {
                gust.dir = blended;
            }
        };

        while(acc >= dt) { 
            updateGust((float)dt);
            simTime += dt;
            Step(ps, emitters, (float)dt, gravity, simTime); 
            acc -= dt; 

            for (size_t i = 0; i < fireworks.size();) {
                Firework& fw = fireworks[i];

                glm::vec3 fwAccel = gravity * 0.8f + glm::vec3(0, 1.5f, 0);
                fw.vel += fwAccel * (float)dt;
                fw.pos += fw.vel * (float)dt;
                fw.fuse -= (float)dt;

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

                if (fw.fuse <= 0.0f || fw.vel.y < 0.0f) {
                    spawnFireworkBurst(ps, fw.pos, 260, 6.0f, 16.0f);

                    fireworks[i] = fireworks.back();
                    fireworks.pop_back();
                    continue;
                }
                i++;
            }
        }

        size_t alive = ps.size(), j = 0; 
        size_t smokeCount = 0, sparkCount = 0, rainCount = 0, snowCount = 0;

        ps.forEachAlive([&](const Particle& p) {
            switch (p.type) {
                case ParticleType::Smoke: smokeCount++; break;
                case ParticleType::Spark: sparkCount++; break;
                case ParticleType::Rain: rainCount++; break;
                case ParticleType::Snow: snowCount++; break;
                default: break;
            }
        });
        size_t baseSmoke = 0;
        size_t baseSnow = baseSmoke + smokeCount;
        size_t baseRain = baseSnow  + snowCount;
        size_t baseSpark = baseRain  + rainCount;

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
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, (baseSpark + sparkCount) * sizeof(DrawParticle), drawBuffer.data());

        // Draw
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   

        glm::mat4 view = cameraView(camera);
        glm::mat4 vp = proj * view;

        glUseProgram(program);
        glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(vp));
        glBindVertexArray(vao);

        glDepthMask(GL_FALSE);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDrawArrays(GL_POINTS, (GLsizei)baseSmoke, (GLsizei)(smokeCount + snowCount + rainCount));

        glBlendFunc(GL_ONE, GL_ONE);
        glDrawArrays(GL_POINTS, (GLsizei)baseSpark, (GLsizei)sparkCount);

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_TRUE);

        {
            const char* patterns[] = { "Plume", "Random", "Rising", "Falling", "Circle" };
            const char* types[] = { "Smoke", "Spark", "Firework", "Rain", "Snow" };
            char title[128];
            snprintf(title, sizeof(title), "Particles | Pattern: %s | Type: %s | Alive: %zu | FPS: %.1f", patterns[(int)currentPattern], types[(int)currentType], alive, fps);
            glfwSetWindowTitle(window, title);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();

        int newWinH, newWinW;
        glfwGetFramebufferSize(window, &newWinW, &newWinH);
        if (newWinW != fbw || newWinH != fbh) {
            fbw = newWinW; fbh = newWinH;
            glViewport(0, 0, fbw, fbh);
            proj = glm::perspective(glm::radians(60.0f), float(fbw)/float(fbh), 0.1f, 200.0f);
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
