FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium + ALL its system dependencies automatically
RUN playwright install --with-deps chromium

# Extra fonts for nice rendering
RUN apt-get update && apt-get install -y \
    fonts-liberation \
    fonts-dejavu-core \
    fonts-noto-core \
    fontconfig \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN mkdir -p output

CMD ["python", "bot.py"]
