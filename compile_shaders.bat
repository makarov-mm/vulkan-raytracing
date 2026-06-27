@echo off
REM Compile all ray tracing shaders to SPIR-V using the Vulkan SDK's glslangValidator.
REM Run automatically as a pre-build step, or manually by double-clicking.

setlocal
cd /d "%~dp0shaders"

set GLSLANG="%VULKAN_SDK%\Bin\glslangValidator.exe"
if not exist %GLSLANG% (
    echo ERROR: glslangValidator not found. Is the Vulkan SDK installed and VULKAN_SDK set?
    exit /b 1
)

echo Compiling shaders with %GLSLANG%

for %%S in (raygen.rgen miss.rmiss shadow.rmiss closesthit.rchit) do (
    %GLSLANG% --target-env vulkan1.2 -V "%%S" -o "%%S.spv"
    if errorlevel 1 (
        echo ERROR compiling %%S
        exit /b 1
    )
    echo   %%S -^> %%S.spv
)

echo Shaders compiled successfully.
endlocal
exit /b 0
