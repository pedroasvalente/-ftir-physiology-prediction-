import subprocess

import typer

app = typer.Typer(add_completion=False)


@app.command()
def train(config: str = typer.Argument(..., help="Path to experiment config JSON")):
    """Run a training experiment from a JSON config file."""
    from ftir_pred.models.training import run_experiment
    run_experiment(config)


@app.command()
def streamlit():
    """Launch the Streamlit dashboard."""
    subprocess.run(["streamlit", "run", "app/main.py"], check=True)


if __name__ == "__main__":
    app()
