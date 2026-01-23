use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use xshell::{cmd, Shell};

#[derive(Parser)]
#[command(name = "xtask")]
#[command(about = "Development automation", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build python bindings and setup environment
    PyBuild {
        /// Recreate venv
        #[arg(long)]
        reset: bool,
    },
    /// Compare PyCoeus with PyTorch
    Compare,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let sh = Shell::new()?;

    match cli.command {
        Commands::PyBuild { reset } => {
            py_build(&sh, reset)?;
        }
        Commands::Compare => {
            run_compare(&sh)?;
        }
    }

    Ok(())
}

fn py_build(sh: &Shell, reset: bool) -> Result<()> {
    let root = project_root();
    sh.change_dir(&root);

    let venv_dir = root.join(".venv");

    // Check if venv exists
    if reset && venv_dir.exists() {
        println!("Removing existing venv...");
        std::fs::remove_dir_all(&venv_dir)?;
    }

    let python = if cfg!(windows) {
        venv_dir.join("Scripts").join("python.exe")
    } else {
        venv_dir.join("bin").join("python")
    };

    if !venv_dir.exists() {
        println!("Creating virtual environment...");
        let venv_str = venv_dir.to_string_lossy().into_owned();
        cmd!(sh, "python -m venv {venv_str}").run()?;
    }

    // Install dependencies
    let python_str = python.to_string_lossy().into_owned();
    println!("Installing dependencies...");
    cmd!(sh, "{python_str} -m pip install maturin torch numpy").run()?;

    // Build with maturin
    println!("Building pycoeus with maturin...");
    // We need to point to pycoeus directory
    cmd!(
        sh,
        "{python_str} -m maturin develop --manifest-path pycoeus/Cargo.toml"
    )
    .run()?;

    println!("Build complete!");
    Ok(())
}

fn run_compare(sh: &Shell) -> Result<()> {
    let root = project_root();
    sh.change_dir(&root);

    let venv_dir = root.join(".venv");
    let python = if cfg!(windows) {
        venv_dir.join("Scripts").join("python.exe")
    } else {
        venv_dir.join("bin").join("python")
    };

    if !python.exists() {
        eprintln!("Virtual environment not found. Please run 'cargo xtask py-build' first.");
        return Ok(());
    }

    let python_str = python.to_string_lossy().into_owned();
    let script_path = "scripts/compare_coeus_torch.py";

    // Make sure script exists
    if !root.join(script_path).exists() {
        eprintln!("Comparison script not found at {}", script_path);
        return Ok(());
    }

    println!("Running comparison script...");
    cmd!(sh, "{python_str} {script_path}").run()?;
    Ok(())
}

fn project_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}
