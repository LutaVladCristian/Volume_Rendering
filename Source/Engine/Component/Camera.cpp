#include <Component/Camera/Camera.h>
#include <Component/Transform/Transform.h>
#include <include/glm.h>
#include <iostream>

using namespace EngineComponents;

Camera::Camera()
{
    transform = new Transform();
    type = CameraType::FirstPerson;
    minSpeed = 1.0f;
    maxSpeed = 5.0f;
    sensitivityOX = 0.1f;
    sensitivityOY = 0.1f;
    limitUp = 89.f;
    limitDown = -89.f;
    FoVy = 60.f;
    aspectRatio = 1.f;
    zNear = 0.01f;
    zFar = 200.f;
    isPerspective = true;
    ortographicWidth = 1.f;
    Update();
}

Camera::~Camera()
{
    delete transform;
}

void Camera::Init() { Update(); }

void Camera::Log() const
{
    glm::vec3 p = transform->GetWorldPosition();
    std::cout << "Camera position: " << p.x << "," << p.y << "," << p.z << std::endl;
}

void Camera::Update()
{
    glm::mat4 model = transform->GetModel();
    View = glm::inverse(model);
}

const glm::mat4& Camera::GetViewMatrix() const { return View; }
const glm::mat4& Camera::GetProjectionMatrix() const { return Projection; }

void Camera::RotateOX(float deltaTime)
{
    transform->RotateWorldOX(deltaTime * sensitivityOX);
}

void Camera::RotateOY(float deltaTime)
{
    transform->RotateWorldOY(deltaTime * sensitivityOY);
}

void Camera::RotateOZ(float deltaTime)
{
    transform->RotateWorldOZ(deltaTime);
}

void Camera::UpdateSpeed(float offset)
{
    float s = transform->GetMoveSpeed();
    s += offset;
    if(s < minSpeed) s = minSpeed;
    if(s > maxSpeed) s = maxSpeed;
    transform->SetMoveSpeed(s);
}

void Camera::SetPosition(const glm::vec3 &position)
{
    transform->SetWorldPosition(position);
    Update();
}

void Camera::SetRotation(const glm::quat &worldRotation)
{
    transform->SetWorldRotation(worldRotation);
    Update();
}

void Camera::SetPositionAndRotation(const glm::vec3 &position, const glm::quat &worldRotation)
{
    transform->SetWorldPosition(position);
    transform->SetWorldRotation(worldRotation);
    Update();
}

void Camera::MoveForward(float deltaTime) { transform->Move( transform->GetLocalOZVector() * (-deltaTime)); Update(); }
void Camera::MoveBackward(float deltaTime) { transform->Move( transform->GetLocalOZVector() * (deltaTime)); Update(); }
void Camera::MoveRight(float deltaTime) { transform->Move( transform->GetLocalOXVector() * (deltaTime)); Update(); }
void Camera::MoveLeft(float deltaTime) { transform->Move( transform->GetLocalOXVector() * (-deltaTime)); Update(); }
void Camera::MoveUp(float deltaTime) { transform->Move( transform->GetLocalOYVector() * (deltaTime)); Update(); }
void Camera::MoveDown(float deltaTime) { transform->Move( transform->GetLocalOYVector() * (-deltaTime)); Update(); }
void Camera::MoveInDirection(glm::vec3 direction, float deltaTime) { transform->Move(direction * deltaTime); Update(); }

void Camera::SetPerspective(float FoVyDeg, float aspectRatio, float zNear, float zFar)
{
    this->FoVy = FoVyDeg;
    this->aspectRatio = aspectRatio;
    this->zNear = zNear;
    this->zFar = zFar;
    isPerspective = true;
    Projection = glm::perspective(glm::radians(FoVyDeg), aspectRatio, zNear, zFar);
}

void Camera::SetOrthographic(float width, float height, float zNear, float zFar)
{
    isPerspective = false;
    Projection = glm::ortho(-width/2.f, width/2.f, -height/2.f, height/2.f, zNear, zFar);
    ortographicWidth = width;
    this->zNear = zNear;
    this->zFar = zFar;
}

void Camera::SetOrthographic(float left, float right, float bottom, float top, float zNear, float zFar)
{
    isPerspective = false;
    Projection = glm::ortho(left, right, bottom, top, zNear, zFar);
    ortographicWidth = right-left;
    this->zNear = zNear;
    this->zFar = zFar;
}

void Camera::SetProjection(const ProjectionInfo &PI)
{
    if(PI.isPerspective)
        SetPerspective(PI.FoVy, PI.aspectRatio, PI.zNear, PI.zFar);
    else
        SetOrthographic(PI.width, PI.height, PI.zNear, PI.zFar);
}

ProjectionInfo Camera::GetProjectionInfo() const
{
    ProjectionInfo p{};
    p.FoVy = FoVy;
    p.zNear = zNear;
    p.zFar = zFar;
    p.aspectRatio = aspectRatio;
    p.width = ortographicWidth;
    p.height = ortographicWidth / aspectRatio;
    p.isPerspective = isPerspective;
    return p;
}

float Camera::GetFieldOfViewY() const { return FoVy; }
float Camera::GetFieldOfViewX() const { return FoVy * aspectRatio; }

void Camera::UpdatePitch(float deltaAngle) { RotateOX(deltaAngle); }
void Camera::SetYaw(float deltaAngle) { RotateOY(deltaAngle); }
void Camera::UpdateRoll(float deltaAngle) { RotateOZ(deltaAngle); }

