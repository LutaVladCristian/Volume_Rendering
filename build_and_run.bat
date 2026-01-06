@echo off
setlocal enabledelayedexpansion

echo =====================================================
echo    Volume Rendering Engine - Build and Run Script
echo =====================================================
echo.

:: Store the project root directory
set PROJECT_ROOT=%~dp0
set PROJECT_ROOT=%PROJECT_ROOT:~0,-1%

:: Set build configuration (Debug or Release)
set BUILD_TYPE=Release
if "%1"=="debug" set BUILD_TYPE=Debug
if "%1"=="Debug" set BUILD_TYPE=Debug

echo Build Configuration: %BUILD_TYPE%
echo Project Root: %PROJECT_ROOT%
echo.

:: Create build directory
if not exist "build" (
    echo Creating build directory...
    mkdir build
)

cd build

:: Run CMake configuration
echo.
echo [1/3] Configuring CMake...
echo =====================================================
cmake -G "Visual Studio 17 2022" -A x64 ..
if %ERRORLEVEL% neq 0 (
    echo.
    echo Trying Visual Studio 2019...
    cmake -G "Visual Studio 16 2019" -A x64 ..
    if %ERRORLEVEL% neq 0 (
        echo.
        echo ERROR: CMake configuration failed!
        echo Please ensure Visual Studio Build Tools are properly installed.
        pause
        exit /b 1
    )
)

:: Build the project
echo.
echo [2/3] Building project (%BUILD_TYPE%)...
echo =====================================================
cmake --build . --config %BUILD_TYPE%
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo [3/3] Build successful! Starting application...
echo =====================================================
echo.

:: Go back to project root to run the executable
:: (shaders are referenced relative to project root)
cd "%PROJECT_ROOT%"

:: Run the executable from project root
if exist "build\bin\VolumeRendering.exe" (
    echo Running VolumeRendering.exe from project root...
    echo.
    "build\bin\VolumeRendering.exe"
) else (
    echo ERROR: Executable not found!
    echo Expected location: build\bin\VolumeRendering.exe
    pause
    exit /b 1
)

endlocal
