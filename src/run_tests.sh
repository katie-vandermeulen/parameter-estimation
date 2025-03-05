#!/usr/bin/env bash

# Set the Python interpreter
PYTHON="python3"

# Ensure Python is installed
if ! command -v $PYTHON &> /dev/null
then
    echo "Error: Python3 is not installed. Please install Python3 and try again."
    exit 1
fi

# Navigate to the root directory of the project (modify if needed)
cd "$(dirname "$0")/.." || { echo "Error: Failed to navigate to project root!"; exit 1; }

# Verify necessary files exist before running tests
REQUIRED_FILES=("src/SimplifiedThreePL.py" "src/Experiment.py" "src/SignalDetection.py" "tests/test_SimplifiedThreePL.py")
for FILE in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$FILE" ]; then
        echo "Error: Required file $FILE not found! Make sure all source files exist."
        exit 1
    fi
done

# Run all unit tests inside the "tests" folder
echo "Running unit and integration tests..."
$PYTHON -m unittest discover -s tests -p "test_*.py"

# Capture the exit status of unittest
TEST_STATUS=$?

# Print success or failure message
if [ $TEST_STATUS -ne 0 ]; then
    echo "Some tests failed! Check test logs for details."
    exit 1
else
    echo "All tests passed successfully!"
    exit 0
fi
