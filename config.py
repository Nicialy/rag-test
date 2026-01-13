from pydantic_settings import BaseSettings
from pydantic_ai.providers.mistral import MistralProvider
from pydantic_ai.models.mistral import MistralModel


class Settings(BaseSettings):

    MODEL: str = 'mistral'
    API_KEY: str
    MISTRAL_MODEL_ID: str = "mistral-large-latest"

    class Config:
        env_file = ".env"


settings = Settings()

provider = MistralProvider(api_key=settings.API_KEY)
model = MistralModel("mistral-large-latest", provider=provider)