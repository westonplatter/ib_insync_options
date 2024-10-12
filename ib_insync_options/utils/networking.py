import platform
import subprocess


def get_ibkr_host_ip() -> str:
    """
    Get the IP address of the host machine. Works for Mac and Linux (WSL on Windows).

    Returns:
        str: The IP address of the host machine.

    """
    os_type = platform.system()

    # mac
    if os_type == "Darwin":
        return "127.0.0.1"

    # If Linux (potentially WSL on Windows), try to find the default gateway IP
    elif os_type == "Linux":
        try:
            # Run the ip route command and decode the output
            output = subprocess.check_output(["ip", "route"]).decode()

            # Find the line that starts with 'default'
            for line in output.splitlines():
                if line.startswith("default"):
                    # Extract the IP address
                    return line.split()[2]
        except Exception as e:
            print(f"Error finding IP address: {e}")
            return None

    else:
        raise NotImplementedError(f"Current OS {os_type} is not supported.")
