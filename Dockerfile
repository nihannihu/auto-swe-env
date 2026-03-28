FROM python:3.11-slim

# 1. Set WORKDIR as root so it gets created successfully
WORKDIR /code

# 2. Create the user and grant ownership of the WORKDIR
RUN useradd -m -u 1000 user
RUN chown -R user /code

# 3. NOW drop privileges to the user mode
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 4. Atomic copy and install
COPY --chown=user ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /code

# 5. Start the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
