import os
from dotenv import load_dotenv


def setup_environment(env_file_name: str = ".env") -> dict:
    """
    /**
     * Loads the environment variables from the .env file in the project root.
     *
     * @param env_file_name The name of the environment file to load.
     * @return A dictionary containing the loaded API credentials.
     * @throws ValueError If the API key or base URL is not found.
     */
    """
    # Dynamically resolve the absolute path to the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    env_path = os.path.join(base_dir, env_file_name)

    load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("HELMHOLTZ_API_KEY")
    api_url = os.getenv("HELMHOLTZ_BASE_URL")

    if not api_key or not api_url:
        raise ValueError(f"Missing API credentials in {env_path}")

    return {
        "api_key": api_key,
        "api_url": api_url
    }