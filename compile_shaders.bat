@echo off
REM Compiles the ray-tracing shaders to SPIR-V with the Vulkan SDK's glslangValidator.
REM Runs as a pre-build step; it can also be started by hand.

setlocal
cd /d "%~dp0shaders"

set GLSLANG="%VULKAN_SDK%\Bin\glslangValidator.exe"
if not exist %GLSLANG% (
    echo ERROR: glslangValidator not found. Is the Vulkan SDK installed and VULKAN_SDK set?
    exit /b 1
)

for %%S in (raygen.rgen miss.rmiss shadow.rmiss closesthit.rchit) do (
    %GLSLANG% --target-env vulkan1.3 -V "%%S" -o "%%S.spv"
    if errorlevel 1 (
        echo ERROR compiling %%S
        exit /b 1
    )
)

echo Shaders compiled successfully.
exit /b 0
