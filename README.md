# Animal Emotion Classification

This project aims to classify animal emotions (happy, sad, angry) from real time camera feed using YOLO and a fine-tuned MobileNetV3 model.

## Project Setup

### 1. Install uv
Ensure that you have a pipx installed.
```bash
pipx install uv
```
Restartthe  shell or IDE.

### 2. Set up Python Environment and Install Dependencies

1.  **Install Python 3.12:**
    ```bash
    uv python install 3.12
    ```

2.  **Install project dependencies:**
    ```bash
    uv sync
    ```

## Running the Application
```bash
uv run python -m src.main
```

## Running Notebooks
Ensure your Jupyter environment (e.g., VS Code, Jupyter Lab) is configured to use the `.venv` kernel created by `uv`.
1. Open the `notebooks/fine_tuning.ipynb` notebook.
2. Select the kernel from `.venv`.
3. Run the cells.

## Running Tests
To run the automated tests, use the following command from the project root:
```bash
uv run pytest
```