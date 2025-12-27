#!/usr/bin/env python3
"""
Prompt Management CLI Tool

Command-line interface for managing agent prompts:
- Export prompts to JSON files for editing
- List all available prompts
- Show detailed prompt information
- Validate prompt configurations
- Compare prompt versions
"""

import argparse
import json
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts import export_prompts, get_all_prompts, get_registry

console = Console()


def cmd_list(args):
    """List all available prompts."""
    prompts = get_all_prompts()

    if args.category:
        prompts = {k: v for k, v in prompts.items() if v.category == args.category}

    # Create table
    table = Table(title="Available Agent Prompts", box=box.ROUNDED)
    table.add_column("Agent Key", style="cyan")
    table.add_column("Agent Name", style="green")
    table.add_column("Version", style="yellow")
    table.add_column("Category", style="magenta")
    table.add_column("Tools", style="blue")

    for key, prompt in sorted(prompts.items()):
        table.add_row(
            key,
            prompt.agent_name,
            prompt.version,
            prompt.category,
            "Yes" if prompt.requires_tools else "No",
        )

    console.print(table)
    console.print(f"\n[bold]Total:[/bold] {len(prompts)} prompts")


def cmd_show(args):
    """Show detailed information about a specific prompt."""
    registry = get_registry()
    prompt = registry.get(args.agent_key)

    if not prompt:
        console.print(f"[red]Error:[/red] No prompt found for key '{args.agent_key}'")
        console.print("\nAvailable keys:")
        for key in registry.get_all().keys():
            console.print(f"  - {key}")
        sys.exit(1)

    # Create detail panel
    details = f"""[bold cyan]Agent Key:[/bold cyan] {prompt.agent_key}
[bold cyan]Agent Name:[/bold cyan] {prompt.agent_name}
[bold cyan]Version:[/bold cyan] {prompt.version}
[bold cyan]Category:[/bold cyan] {prompt.category}
[bold cyan]Requires Tools:[/bold cyan] {prompt.requires_tools}

[bold cyan]System Message:[/bold cyan]
{prompt.system_message}
"""

    if prompt.metadata:
        details += f"\n[bold cyan]Metadata:[/bold cyan]\n{json.dumps(prompt.metadata, indent=2)}"

    panel = Panel(
        details, title=f"Prompt Details: {prompt.agent_name}", border_style="cyan"
    )
    console.print(panel)


def cmd_export(args):
    """Export all prompts to JSON files."""
    output_dir = Path(args.output_dir)

    try:
        export_prompts(str(output_dir))
        console.print(f"[green]Success![/green] Prompts exported to: {output_dir}")
        console.print(
            f"\nYou can now edit the JSON files in '{output_dir}' to customize prompts."
        )
        console.print(
            "The system will automatically load your customized prompts on next run."
        )
    except Exception as e:
        console.print(f"[red]Error exporting prompts:[/red] {e}")
        sys.exit(1)


def cmd_validate(args):
    """Validate all prompts."""
    registry = get_registry()
    issues = registry.validate()

    if not issues:
        console.print("[green]All prompts are valid![/green]")
        return

    console.print(f"[yellow]Found {len(issues)} validation issues:[/yellow]\n")
    for issue in issues:
        console.print(f"  [red]✗[/red] {issue}")

    sys.exit(1)


def cmd_categories(args):
    """List all prompt categories."""
    prompts = get_all_prompts()
    categories = {}

    for prompt in prompts.values():
        if prompt.category not in categories:
            categories[prompt.category] = []
        categories[prompt.category].append(prompt.agent_key)

    table = Table(title="Prompt Categories", box=box.ROUNDED)
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="yellow")
    table.add_column("Agent Keys", style="green")

    for category, keys in sorted(categories.items()):
        table.add_row(category, str(len(keys)), ", ".join(sorted(keys)))

    console.print(table)


def cmd_compare(args):
    """Compare two prompt versions."""
    if not args.file1.exists() or not args.file2.exists():
        console.print("[red]Error:[/red] One or both files not found")
        sys.exit(1)

    try:
        with open(args.file1) as f:
            prompt1 = json.load(f)
        with open(args.file2) as f:
            prompt2 = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing JSON:[/red] {e}")
        sys.exit(1)

    # Compare
    console.print("\n[bold]Comparing:[/bold]")
    console.print(f"  File 1: {args.file1} (v{prompt1.get('version', 'unknown')})")
    console.print(f"  File 2: {args.file2} (v{prompt2.get('version', 'unknown')})")
    console.print()

    differences = []

    for key in ["agent_key", "agent_name", "version", "category", "requires_tools"]:
        val1 = prompt1.get(key)
        val2 = prompt2.get(key)
        if val1 != val2:
            differences.append((key, val1, val2))

    # Check system message
    msg1 = prompt1.get("system_message", "")
    msg2 = prompt2.get("system_message", "")
    if msg1 != msg2:
        differences.append(
            ("system_message", f"{len(msg1)} chars", f"{len(msg2)} chars")
        )
        console.print("[yellow]System messages differ:[/yellow]")
        console.print(f"\n[cyan]File 1 message:[/cyan]\n{msg1}\n")
        console.print(f"[cyan]File 2 message:[/cyan]\n{msg2}\n")

    if differences:
        table = Table(title="Differences", box=box.ROUNDED)
        table.add_column("Field", style="cyan")
        table.add_column("File 1", style="yellow")
        table.add_column("File 2", style="green")

        for field, val1, val2 in differences:
            if field != "system_message":
                table.add_row(field, str(val1), str(val2))

        console.print(table)
    else:
        console.print("[green]No differences found (excluding metadata)[/green]")


def cmd_init(args):
    """Initialize prompts directory with default prompts."""
    output_dir = Path(args.output_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.force:
            console.print(
                f"[yellow]Warning:[/yellow] Directory '{output_dir}' already exists and is not empty."
            )
            console.print("Use --force to overwrite existing files.")
            sys.exit(1)

    try:
        export_prompts(str(output_dir))
        console.print(f"[green]Success![/green] Initialized prompts in: {output_dir}")
        console.print("\nNext steps:")
        console.print("  1. Edit JSON files to customize prompts")
        console.print("  2. Update version numbers when you modify prompts")
        console.print("  3. Run: python scripts/prompt_manager.py validate")
        console.print(
            "  4. Run your analysis - custom prompts will be loaded automatically"
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Manage agent prompts for the Multi-Agent Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize prompts directory
  python scripts/prompt_manager.py init

  # List all prompts
  python scripts/prompt_manager.py list

  # Show specific prompt
  python scripts/prompt_manager.py show market_analyst

  # Export prompts for editing
  python scripts/prompt_manager.py export

  # Validate prompts
  python scripts/prompt_manager.py validate

  # Compare two versions
  python scripts/prompt_manager.py compare prompts/market_analyst.json prompts/market_analyst.v2.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List all available prompts")
    list_parser.add_argument("--category", help="Filter by category")
    list_parser.set_defaults(func=cmd_list)

    # Show command
    show_parser = subparsers.add_parser("show", help="Show detailed prompt information")
    show_parser.add_argument("agent_key", help="Agent key to show")
    show_parser.set_defaults(func=cmd_show)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export prompts to JSON files")
    export_parser.add_argument(
        "--output-dir",
        default="./prompts",
        help="Output directory (default: ./prompts)",
    )
    export_parser.set_defaults(func=cmd_export)

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize prompts directory")
    init_parser.add_argument(
        "--output-dir",
        default="./prompts",
        help="Output directory (default: ./prompts)",
    )
    init_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing files"
    )
    init_parser.set_defaults(func=cmd_init)

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate all prompts")
    validate_parser.set_defaults(func=cmd_validate)

    # Categories command
    categories_parser = subparsers.add_parser(
        "categories", help="List prompt categories"
    )
    categories_parser.set_defaults(func=cmd_categories)

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare", help="Compare two prompt versions"
    )
    compare_parser.add_argument("file1", type=Path, help="First prompt file")
    compare_parser.add_argument("file2", type=Path, help="Second prompt file")
    compare_parser.set_defaults(func=cmd_compare)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
