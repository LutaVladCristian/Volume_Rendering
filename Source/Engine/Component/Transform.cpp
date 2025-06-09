#include <Component/Transform/Transform.h>
#include <include/glm.h>

using namespace EngineComponents;

Transform::Transform()
{
    _worldPosition = glm::vec3(0.f);
    _localPosition = glm::vec3(0.f);
    _worldRotation = glm::quat();
    _relativeRotation = glm::quat();
    _invWorldRotation = glm::quat();
    _localScale = glm::vec3(1.f);
    _rotateSpeed = 1.f;
    _moveSpeed = 1.f;
    _scaleSpeed = 1.f;
    _motionState = false;
    _modelIsOutdated = true;
    _updateHierarchy = false;
    _parentNode = nullptr;
}

Transform::Transform(const Transform& t)
{
    *this = t;
}

Transform::~Transform() = default;

void Transform::SetHierarchyUpdate(bool value) { _updateHierarchy = value; }
void Transform::ClearMotionState() { _motionState = false; }
bool Transform::GetMotionState() const { return _motionState; }

glm::vec3 Transform::GetLocalPosition() const { return _localPosition; }
glm::vec3 Transform::GetWorldPosition() const { return _worldPosition; }
glm::quat Transform::GetWorldRotation() const { return _worldRotation; }
glm::quat Transform::GetRelativeRotation() const { return _relativeRotation; }
glm::vec3 Transform::GetRotationEulerRad() const { return glm::eulerAngles(_worldRotation); }
glm::vec3 Transform::GetRotationEuler360() const { return glm::degrees(glm::eulerAngles(_worldRotation)); }

glm::vec3 Transform::GetLocalOXVector() const { return _worldRotation * glm::vec3(1,0,0); }
glm::vec3 Transform::GetLocalOYVector() const { return _worldRotation * glm::vec3(0,1,0); }
glm::vec3 Transform::GetLocalOZVector() const { return _worldRotation * glm::vec3(0,0,1); }

glm::vec3 Transform::GetScale() const { return _localScale; }

const glm::mat4& Transform::GetModel()
{
    if (_modelIsOutdated)
        ComputeWorldModel();
    return _worldModel;
}

float Transform::GetMoveSpeed() const { return _moveSpeed; }
float Transform::GetScaleSpeed() const { return _scaleSpeed; }
float Transform::GetRotationSpeed() const { return _rotateSpeed; }

void Transform::Move(const glm::vec3 &offset)
{
    _worldPosition += offset;
    _modelIsOutdated = true;
}

void Transform::Move(const glm::vec3 &dir, float deltaTime)
{
    _worldPosition += dir * _moveSpeed * deltaTime;
    _modelIsOutdated = true;
}

void Transform::Scale(float deltaTime)
{
    _localScale += glm::vec3(1.f) * _scaleSpeed * deltaTime;
    _modelIsOutdated = true;
}

void Transform::RotateWorldOX(float deltaTime) { _worldRotation = glm::rotate(_worldRotation, glm::radians(deltaTime * _rotateSpeed), glm::vec3(1,0,0)); _modelIsOutdated = true; }
void Transform::RotateWorldOY(float deltaTime) { _worldRotation = glm::rotate(_worldRotation, glm::radians(deltaTime * _rotateSpeed), glm::vec3(0,1,0)); _modelIsOutdated = true; }
void Transform::RotateWorldOZ(float deltaTime) { _worldRotation = glm::rotate(_worldRotation, glm::radians(deltaTime * _rotateSpeed), glm::vec3(0,0,1)); _modelIsOutdated = true; }
void Transform::RotateLocalOX(float deltaTime) { RotateWorldOX(deltaTime); }
void Transform::RotateLocalOY(float deltaTime) { RotateWorldOY(deltaTime); }
void Transform::RotateLocalOZ(float deltaTime) { RotateWorldOZ(deltaTime); }

void Transform::SetLocalPosition(glm::vec3 position) { _localPosition = position; _worldPosition = position; _modelIsOutdated = true; }
void Transform::SetWorldPosition(glm::vec3 position) { _worldPosition = position; _modelIsOutdated = true; }

void Transform::SetWorldRotation(glm::quat rotationQ) { _worldRotation = rotationQ; _modelIsOutdated = true; }
void Transform::SetWorldRotation(const glm::vec3 &eulerAngles360) { _worldRotation = glm::quat(glm::radians(eulerAngles360)); _modelIsOutdated = true; }
void Transform::SetWorldRotationAndScale(const glm::quat &rotationQ, glm::vec3 scale) { _worldRotation = rotationQ; _localScale = scale; _modelIsOutdated = true; }

void Transform::SetReleativeRotation(const glm::vec3 &eulerAngles360) { _relativeRotation = glm::quat(glm::radians(eulerAngles360)); _modelIsOutdated = true; }
void Transform::SetReleativeRotation(const glm::quat &localRotationQ) { _relativeRotation = localRotationQ; _modelIsOutdated = true; }

void Transform::SetScale(glm::vec3 scale) { _localScale = scale; _modelIsOutdated = true; }
void Transform::ForceUpdate() { ComputeWorldModel(); }

void Transform::Copy(const Transform &source) { *this = source; }
void Transform::SetMoveSpeed(float unitsPerSecond) { _moveSpeed = unitsPerSecond; }
void Transform::SetScaleSpeed(float unitsPerSecond) { _scaleSpeed = unitsPerSecond; }
void Transform::SetRotationSpeed(float degreesPerSecond) { _rotateSpeed = degreesPerSecond; }

void Transform::AddChild(Transform *transform) { if(transform) _childNodes.push_back(transform); }
void Transform::RemoveChild(Transform *transform) { _childNodes.remove(transform); }

float Transform::DistanceTo(Transform *transform) { return glm::length(GetWorldPosition() - transform->GetWorldPosition()); }
float Transform::DistanceTo(const glm::vec3 &position) { return glm::length(GetWorldPosition() - position); }
float Transform::Distance2To(Transform *transform) { glm::vec3 d = GetWorldPosition() - transform->GetWorldPosition(); return glm::dot(d,d); }
float Transform::Distance2To(const glm::vec3 &position) { glm::vec3 d = GetWorldPosition() - position; return glm::dot(d,d); }

glm::vec3 Transform::GetRelativePositionOf(const Transform &transform) {
    return glm::inverse(glm::mat3_cast(_worldRotation)) * (transform.GetWorldPosition() - _worldPosition);
}

void Transform::Init() {}
void Transform::ComputeWorldModel() { _worldModel = glm::translate(glm::mat4(1.f), _worldPosition) * glm::toMat4(_worldRotation) * glm::scale(glm::mat4(1.f), _localScale); _modelIsOutdated=false; }
void Transform::UpdateWorldModel() { ComputeWorldModel(); }
void Transform::UpdateWorldPosition() {}
void Transform::UpdateLocalPosition() {}
void Transform::UpdateRelativeRotation() {}
void Transform::UpdateWorldInfo() {}
void Transform::UpdateChildsPosition() {}
void Transform::UpdateChildrenRotation() {}
void Transform::UpdateModelPosition() {}

