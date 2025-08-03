@echo off
echo ========================================
echo Dihedral Analyzer v1.3 - Local Build
echo ========================================

echo.
echo Installing dependencies...
pip install MDAnalysis numpy scipy matplotlib tqdm pyinstaller

echo.
echo Building executable with custom spec file...
pyinstaller dihedral_analyzer_1.3_windows.spec

echo.
echo Testing executable...
if exist "dist\dihedral_analyzer_1.3_windows.exe" (
    echo Testing --help command...
    dist\dihedral_analyzer_1.3_windows.exe --help
    echo.
    echo Build completed successfully!
    echo Executable: dist\dihedral_analyzer_1.3_windows.exe
) else (
    echo ERROR: Executable not found!
    echo Check the build logs above for errors.
)

echo.
echo Press any key to exit...
pause > nul 