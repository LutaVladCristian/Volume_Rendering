#include "InputController.h"

#include "../Engine.h"

InputController::InputController()
{
	window = Engine::GetWindow();
	window->SubscribeToEvents(this);
	isAttached = true;
}

InputController::~InputController()
{
	if (isAttached && window)
		window->UnsubscribeFromEvents(this);
}

bool InputController::IsActive() const
{
	return isAttached;
}

void InputController::SetActive(bool value)
{
	isAttached = value;
	value ? window->SubscribeToEvents(this) : window->UnsubscribeFromEvents(this);
}
