from pydantic import BaseModel


class ForecastRequest(BaseModel):
    item_id: str
    history: list  # historical sales (e.g., [120, 130, 128, 135, ...])
    steps_ahead: int = 7  # how many days to forecast (default = 7)