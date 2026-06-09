import typer
from pathlib import Path
from preprocessing_at_startup import build_cis_benchmark_vector_db, build_tenant_collection
from supervisor_agent import start_agent

app = typer.Typer()

def read_file(source_path: Path) -> str:
    ext = source_path.suffix.lower()

    if ext == ".pdf":
        import pdfplumber
        with pdfplumber.open(source_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    elif ext == ".docx":
        from docx import Document
        doc = Document(source_path)
        return "\n".join(p.text for p in doc.paragraphs)

    elif ext in (".xlsx", ".csv"):
        import pandas as pd
        df = pd.read_excel(source_path) if ext == ".xlsx" else pd.read_csv(source_path)
        return df.to_string()

    else:
        return source_path.read_text(encoding="utf-8")


@app.command()
def start_app(
    source: Path = typer.Argument(..., help="Path to the security policy file"),
):
    """Load a security policy file, convert it to .txt, and store it in the app."""

    if not source.exists():
        typer.echo(f"Error: File not found at '{source}'")
        raise typer.Exit(code=1)

    if not source.is_file():
        typer.echo(f"Error: '{source}' is not a file")
        raise typer.Exit(code=1)

    typer.echo("Reading file...")
    contents = read_file(source)

    # Filter out empty lines, then join with \n\n
    lines = [line.strip() for line in contents.splitlines() if line.strip()]
    formatted_contents = "\n\n".join(lines)

    dest_path = Path("graphtutorial/agents/security_policy.txt")
    dest_path.write_text(formatted_contents, encoding="utf-8")

    typer.secho(f"\n✅ Saved to: {dest_path}", fg=typer.colors.GREEN)
    typer.echo(f"📄 Size: {len(formatted_contents)} characters")
    typer.echo("\n--- Preview ---")
    typer.echo(formatted_contents[:200])

    typer.echo("\nNow that we have loaded the policy, let's analyze the tenant's configuration.\n")
    build_cis_benchmark_vector_db(force_rebuild=False)
    typer.echo("CIS benchmark data processed and vector database built.\n")
    build_tenant_collection(platform="windows", force_rebuild=False)
    typer.echo("Tenant configuration data processed and vector database built.\n")
    start_agent()


if __name__ == "__main__":
    typer.echo("Welcome to the Microsoft Graph Security Posture Agent!\n")
    app()
    
