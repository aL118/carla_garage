#!/usr/bin/env bash

# Default directories
DEFAULT_DOWNLOAD_DIR="../data"
DEFAULT_EXTRACT_DIR="../data"

# Parse command-line arguments
DOWNLOAD_DIR="${1:-$DEFAULT_DOWNLOAD_DIR}"
EXTRACT_DIR="${2:-$DEFAULT_EXTRACT_DIR}"

# Display usage if help is requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [DOWNLOAD_DIR] [EXTRACT_DIR]"
    echo "  DOWNLOAD_DIR: Directory where tar files will be downloaded (default: $DEFAULT_DOWNLOAD_DIR)"
    echo "  EXTRACT_DIR: Directory where files will be extracted (default: $DEFAULT_EXTRACT_DIR)"
    echo "Example: $0 /tmp/downloads /home/user/extracted_data"
    exit 0
fi

# Create directories if they don't exist
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$EXTRACT_DIR"

down_load_unzip() {
    local scenario="$1"
    local download_path="$DOWNLOAD_DIR/${scenario}.tar"

    # Download to specified directory
    wget -O "$download_path" "https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/dataset/${scenario}.tar"

    # Extract to specified directory
    tar -xf "$download_path" -C "$EXTRACT_DIR"

    # Remove the downloaded tar file to free up space
    rm "$download_path"
}

# Download subset of scenarios (~200GB total)
# Selected 21 most important scenarios for training
for scenario in Accident AccidentTwoWays BlockedIntersection DynamicObjectCrossing EnterActorFlow HardBreakRoute HazardAtSideLane HighwayCutIn HighwayExit InterurbanActorFlow InvadingTurn MergerIntoSlowTraffic NonSignalizedJunctionLeftTurn NonSignalizedJunctionRightTurn OppositeVehicleRunningRedLight ParkedObstacle PedestrianCrossing PriorityAtJunction SignalizedJunctionLeftTurn SignalizedJunctionRightTurn StaticCutIn
do
    down_load_unzip "${scenario}" &
done

wait  # Wait for all background processes to complete
