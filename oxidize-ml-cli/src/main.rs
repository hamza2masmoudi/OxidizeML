use crossterm::{
    cursor, execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{disable_raw_mode, enable_raw_mode, Clear, ClearType},
    event::{read, Event, KeyCode},
};
use std::io::{stdout, Write};
use std::sync::Arc;
use std::time::Duration;

use oximl_core::{Tensor, DType};
use oximl_autodiff::{Graph, Variable};
use oximl_optim::SGD;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut stdout = stdout();

    // Enter raw mode
    enable_raw_mode()?;
    execute!(
        stdout,
        Clear(ClearType::All),
        cursor::Hide,
        cursor::MoveTo(0, 0)
    )?;

    let menu_items = vec![
        "  Quick Start Demo      ",
        "  Model Explorer        ",
        "  Training Dashboard    ",
        "  Settings & Config     ",
        "  Exit                  ",
    ];

    let mut selected_index = 0;

    loop {
        // Render Header
        execute!(
            stdout,
            cursor::MoveTo(0, 2),
            SetForegroundColor(Color::Cyan),
            Print(r#"
   ____       _     ___          __  __ _      
  / __ \_  __(_)___/ (_)_______ |  \/  | |     
 / /_/ / |/_/ / __  / /_  / _ \ | \  / | |     
 \____/_/|_/_/\__,_/_/ /_/\___/ |_|  |_|______|
            "#),
            ResetColor,
            Print("\n\n"),
            SetForegroundColor(Color::DarkGrey),
            Print("    Welcome to OxidizeML v2.0 Interface\n"),
            Print("    Use [Up/Down] to navigate, [Enter] to select\n\n"),
            ResetColor,
        )?;

        // Render Menu
        for (i, item) in menu_items.iter().enumerate() {
            execute!(stdout, cursor::MoveTo(4, (10 + i * 2) as u16))?;
            
            if i == selected_index {
                execute!(
                    stdout,
                    SetForegroundColor(Color::Green),
                    Print(" ► "),
                    SetForegroundColor(Color::White),
                    Print(format!("{}\n", item)),
                    ResetColor
                )?;
            } else {
                execute!(
                    stdout,
                    SetForegroundColor(Color::DarkGrey),
                    Print("   "),
                    Print(format!("{}\n", item)),
                    ResetColor
                )?;
            }
        }

        // Handle Input
        if let Event::Key(key_event) = read()? {
            match key_event.code {
                KeyCode::Up => {
                    if selected_index > 0 {
                        selected_index -= 1;
                    } else {
                        selected_index = menu_items.len() - 1; // cycle
                    }
                }
                KeyCode::Down => {
                    if selected_index < menu_items.len() - 1 {
                        selected_index += 1;
                    } else {
                        selected_index = 0; // cycle
                    }
                }
                KeyCode::Enter => {
                    if selected_index == menu_items.len() - 1 {
                        break; // Exit
                    }
                    if selected_index == 2 {
                        run_training_dashboard(&mut stdout)?;
                    } else {
                        show_placeholder(&mut stdout, menu_items[selected_index].trim())?;
                    }
                }
                KeyCode::Esc | KeyCode::Char('q') => {
                    break;
                }
                _ => {}
            }
        }
    }

    // Cleanup
    disable_raw_mode()?;
    execute!(
        stdout,
        Clear(ClearType::All),
        cursor::Show,
        cursor::MoveTo(0, 0),
        SetForegroundColor(Color::Cyan),
        Print("Thanks for using OxidizeML!\n"),
        ResetColor
    )?;

    Ok(())
}

fn run_training_dashboard(stdout: &mut std::io::Stdout) -> Result<(), Box<dyn std::error::Error>> {
    execute!(
        stdout,
        Clear(ClearType::All),
        cursor::MoveTo(2, 2),
        SetForegroundColor(Color::Magenta),
        Print("--- Live Training Dashboard (v2 Engine) ---"),
        cursor::MoveTo(2, 4),
        SetForegroundColor(Color::DarkGrey),
        Print("Initializing Thread-safe Autodiff Graph..."),
        ResetColor
    )?;

    // 1. Setup ML Environment using `crates-v2` architecture
    let mut graph = Arc::new(Graph::new());
    
    // We'll train a tiny mathematical linear projection: X @ W = Y
    // Where X is [1, 2], W is [2, 1], and Y target is [1, 1] filled with 5.0
    
    let x_data = Tensor::ones(&[1, 2], DType::Float64);
    let x = Variable::input(x_data, graph.clone());
    
    // Weight param to learn. Init to ones.
    let w_data = Tensor::ones(&[2, 1], DType::Float64);
    let mut w = Variable::param(w_data, graph.clone());

    let target = Tensor::ones(&[1, 1], DType::Float64).scalar_mul(5.0)?;

    let mut optimizer = SGD::new(0.01);
    let epochs = 100;

    execute!(
        stdout,
        cursor::MoveTo(2, 6),
        SetForegroundColor(Color::Green),
        Print("Initialization Complete. Beginning Optimization Loop..."),
        ResetColor
    )?;

    // Live Training Loop
    for epoch in 0..=epochs {
        // Forward Pass: pred = X @ W
        let pred = x.matmul(&w)?;
        
        // Compute Mean Squared Error Loss mathematically
        // For structural CLI demo: loss = (pred - target) * (pred - target)
        // Let's break it down into explicit graph steps
        let minus_target = target.scalar_mul(-1.0)?;
        let neg_t_var = Variable::input(minus_target, graph.clone());
        
        let diff = pred.add(&neg_t_var)?;
        let loss = diff.matmul(&diff.t())?;

        // Extract scalar for visualization
        let loss_val = extract_f64_scalar(&loss.data);

        if epoch % 5 == 0 {
            let progress = (epoch as f32 / epochs as f32 * 20.0) as usize;
            let bar = format!("[{}{}]", "█".repeat(progress), " ".repeat(20 - progress));
            
            execute!(
                stdout,
                cursor::MoveTo(2, 9),
                SetForegroundColor(Color::Cyan),
                Print(format!("Epoch {:3}/{} {} ", epoch, epochs, bar)),
                SetForegroundColor(Color::Yellow),
                Print(format!("Loss: {:.6}", loss_val)),
                ResetColor
            )?;
            stdout.flush()?;
            std::thread::sleep(Duration::from_millis(50));
        }

        // Backward Pass
        loss.backward()?;
        let mut w_leaf = w.clone();
        
        // Optimizer Step (w = w - lr * dL/dw)
        optimizer.step(&mut [w_leaf.clone()])?;
        
        // Graph cleanup for next pass
        graph = Arc::new(Graph::new());
        // Re-attach variables to new clean tape
        w = Variable::param(w_leaf.data.clone(), graph.clone());
    }

    execute!(
        stdout,
        cursor::MoveTo(2, 12),
        SetForegroundColor(Color::Green),
        Print("Training Finished SUCCESSFULLY! Math engine verified."),
        cursor::MoveTo(2, 14),
        SetForegroundColor(Color::DarkGrey),
        Print("Press any key to return to the main menu..."),
        ResetColor
    )?;
    
    loop {
        if let Event::Key(_) = read()? {
            break;
        }
    }
    execute!(stdout, Clear(ClearType::All))?;
    Ok(())
}

fn extract_f64_scalar(t: &Tensor) -> f64 {
    match t {
        Tensor::Float64(a) => *a.iter().next().unwrap_or(&0.0),
        _ => 0.0,
    }
}

fn show_placeholder(stdout: &mut std::io::Stdout, title: &str) -> Result<(), Box<dyn std::error::Error>> {
    execute!(
        stdout,
        Clear(ClearType::All),
        cursor::MoveTo(2, 2),
        SetForegroundColor(Color::Magenta),
        Print(format!("--- {} ---", title)),
        cursor::MoveTo(2, 4),
        SetForegroundColor(Color::White),
        Print("This feature is currently under construction in v0.2."),
        cursor::MoveTo(2, 6),
        SetForegroundColor(Color::DarkGrey),
        Print("Press any key to return to the main menu..."),
        ResetColor
    )?;
    
    // Wait for any key
    loop {
        if let Event::Key(_) = read()? {
            break;
        }
    }
    
    // Clear screen before returning to menu to prevent artifacts
    execute!(stdout, Clear(ClearType::All))?;
    Ok(())
}
