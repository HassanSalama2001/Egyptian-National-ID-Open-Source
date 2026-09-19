import typer
import cv2
import json
import numpy as np
import os
from pathlib import Path
from PIL import Image, ImageOps
from rich.console import Console
from rich.table import Table

# Add local path logic if needed or assume installed
from .core.pipeline import Pipeline
from .core.config import settings

app = typer.Typer(help="Egyptian National ID OCR extraction tool")
console = Console()

@app.command()
def extract(
    image_path: Path = typer.Argument(..., help="Path to the National ID image file"),
    output: str = typer.Option("table", "--output", "-o", help="Output format (table, json)"),
    include_images: bool = typer.Option(
        False,
        "--include-images/--no-include-images",
        help=(
            "Include base64 image payloads (aligned card + per-field crops) "
            "in JSON output. Off by default - they run to over a megabyte, "
            "which is rarely what you want in a terminal or piped into jq."
        ),
    ),
):
    """Extract data from an Egyptian National ID card image.

    Note the default differs from the HTTP API's on purpose: the API
    serves a web UI that displays the crops, so it includes them unless
    asked not to. A CLI writes to a terminal or a pipe, where a megabyte
    of base64 is noise, so here they are opt-in.
    """
    if not image_path.exists():
        console.print(f"[red]Error: File {image_path} not found.[/red]")
        raise typer.Exit(1)
        
    # PIL + exif_transpose, not cv2.imread: cv2 ignores EXIF Orientation, so
    # a real phone photo could load sideways/mirrored - see src/app.py for
    # the full explanation.
    try:
        pil_image = Image.open(image_path)
        pil_image = ImageOps.exif_transpose(pil_image)
        image = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        image = None
    if image is None:
        console.print("[red]Error: Could not decode image.[/red]")
        raise typer.Exit(1)
        
    pipeline = Pipeline(tesseract_cmd=settings.TESSERACT_CMD)
    
    with console.status("[bold green]Processing image..."):
        result = pipeline.process_image(image)
        
    if not include_images:
        result = result.without_images()

    if output == "json":
        console.print(result.model_dump_json(indent=2))
    else:
        # Display as table
        table = Table(title="Extracted ID Data")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="magenta")

        if result.front:
            table.add_row("Side", "Front")
            table.add_row("First Name", result.front.first_name)
            table.add_row("First Name (EN)", result.front.first_name_english)
            table.add_row("Full Name", result.front.full_name)
            table.add_row("Full Name (EN)", result.front.full_name_english)
            table.add_row("Address", result.front.address)
            table.add_row("Address (EN)", result.front.address_english)
            table.add_row("National ID", result.front.national_id)
            table.add_row("Date of Birth", result.front.date_of_birth)
            table.add_row("Serial Number", result.front.card_serial_number)

        # The back side used to print nothing at all here beyond the
        # timing row - the table only ever handled the front.
        if result.back:
            table.add_row("Side", "Back")
            table.add_row("National ID", result.back.national_id)
            table.add_row("Issue Date", result.back.issue_date)
            table.add_row("Expiry Date", result.back.expiry_date)
            table.add_row("Profession", result.back.profession)
            table.add_row("Profession (EN)", result.back.profession_english)
            table.add_row("Gender", result.back.gender.value)
            table.add_row("Religion", result.back.religion.value)
            table.add_row("Marital Status", result.back.marital_status.value)

        if result.decoded:
            table.add_row("---", "---")
            table.add_row("Decoded DOB", str(result.decoded.birth_date))
            table.add_row("Decoded Gender", result.decoded.gender.value)
            table.add_row("Decoded Gov", result.decoded.governorate.value)
            
        table.add_row("---", "---")
        table.add_row("Status", result.status.value)
        table.add_row("Confidence", f"{result.confidence:.0%}")
        table.add_row("Time Taken", f"{result.processing_time_ms}ms")

        console.print(table)

        # The status messages carry the "this needs review and here's why"
        # detail that a bare confidence number doesn't convey.
        for message in result.messages:
            console.print(f"[yellow]{message}[/yellow]")

@app.command()
def doctor():
    """Verify system dependencies (PaddleOCR, OpenCV, digit classifier model, etc.)"""
    console.print("[bold cyan]Egyptian National ID OCR — System Check[/bold cyan]")

    all_ok = True

    # 1. Core CV/ML dependencies
    try:
        import cv2
        console.print(f"[OK] OpenCV found: [green]{cv2.__version__}[/green]")
    except ImportError:
        console.print("[FAIL] [red]OpenCV not found.[/red] Run: pip install opencv-python-headless")
        all_ok = False

    # 2. PaddleOCR - the default engine for name/address (see
    # ocr/paddle_ocr_engine.py)
    try:
        import paddleocr
        console.print(f"[OK] PaddleOCR found: [green]{paddleocr.__version__}[/green]")
    except ImportError:
        console.print("[FAIL] [red]PaddleOCR not found.[/red] Run: pip install -e .")
        all_ok = False

    # 3. Digit classifier model (national_id/serial_number/birth_date)
    from pathlib import Path
    model_path = Path(__file__).parent / "ocr" / "models" / "digit_classifier_svm.joblib"
    if model_path.exists():
        console.print(f"[OK] Digit classifier model found: [green]{model_path.name}[/green]")
    else:
        console.print(
            "[FAIL] [red]Digit classifier model not found.[/red] "
            "Train it: python scripts/training/train_digit_classifier.py"
        )
        all_ok = False

    # 4. Tesseract - optional, only used by the (non-default) TesseractEngine
    tess_path = settings.TESSERACT_CMD
    import shutil
    is_found = shutil.which(tess_path) or (os.path.exists(tess_path) if os.name == 'nt' else False)
    if is_found:
        console.print(f"[OK] Tesseract found (optional, not the default engine): [green]{tess_path}[/green]")
    else:
        console.print("[dim]Tesseract not found - fine, it's an optional alternate engine, not the default.[/dim]")

    console.print()
    if all_ok:
        console.print("[bold green]Ready to initiate scan sessions![/bold green]")
    else:
        console.print("[bold red]Some required dependencies are missing - see above.[/bold red]")

if __name__ == "__main__":
    app()
