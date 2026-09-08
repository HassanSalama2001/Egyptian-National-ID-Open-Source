import typer
import cv2
import json
import os
from pathlib import Path
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
):
    """Extract data from an Egyptian National ID card image."""
    if not image_path.exists():
        console.print(f"[red]Error: File {image_path} not found.[/red]")
        raise typer.Exit(1)
        
    image = cv2.imread(str(image_path))
    if image is None:
        console.print("[red]Error: Could not decode image.[/red]")
        raise typer.Exit(1)
        
    pipeline = Pipeline(tesseract_cmd=settings.TESSERACT_CMD)
    
    with console.status("[bold green]Processing image..."):
        result = pipeline.process_image(image)
        
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
            table.add_row("Full Name", result.front.full_name)
            table.add_row("Address", result.front.address)
            table.add_row("National ID", result.front.national_id)
            table.add_row("Serial Number", result.front.card_serial_number)
            
        if result.decoded:
            table.add_row("---", "---")
            table.add_row("Decoded DOB", str(result.decoded.birth_date))
            table.add_row("Decoded Gender", result.decoded.gender.value)
            table.add_row("Decoded Gov", result.decoded.governorate.value)
            
        table.add_row("---", "---")
        table.add_row("Time Taken", f"{result.processing_time_ms}ms")
        
        console.print(table)

@app.command()
def doctor():
    """Verify system dependencies (Tesseract, OpenCV, etc.)"""
    console.print("[bold cyan]Egyptian National ID OCR — System Check[/bold cyan]")
    
    # 1. Check Tesseract
    tess_path = settings.TESSERACT_CMD
    import shutil
    is_found = shutil.which(tess_path) or (os.path.exists(tess_path) if os.name == 'nt' else False)
    
    if is_found:
        console.print(f"[OK] Tesseract found: [green]{tess_path}[/green]")
    else:
        console.print("[FAIL] [red]Tesseract Not Found![/red]")
        console.print("\n[bold yellow]How to install Tesseract:[/bold yellow]")
        console.print("  - [bold]Windows:[/bold] Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        console.print("  - [bold]Linux:[/bold] sudo apt install tesseract-ocr tesseract-ocr-ara")
        console.print("  - [bold]Mac:[/bold] brew install tesseract tesseract-lang")
        console.print("\n[dim]Note: Ensure you include the 'Arabic' language data during installation.[/dim]")

    # 2. Check Python Dependencies
    try:
        import cv2
        console.print(f"[OK] OpenCV found: [green]{cv2.__version__}[/green]")
    except ImportError:
        console.print("[FAIL] [red]OpenCV not found.[/red] Run: pip install opencv-python-headless")

    try:
        import easyocr
        console.print("[OK] EasyOCR found")
    except ImportError:
        console.print("[WARN] [yellow]EasyOCR not found (Optional fallback).[/yellow]")

    console.print("\n[bold]Ready to initiate scan sessions![/bold]")

if __name__ == "__main__":
    app()
