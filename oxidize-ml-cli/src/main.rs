use crossterm::{
    cursor, execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{disable_raw_mode, enable_raw_mode, Clear, ClearType},
    event::{read, Event, KeyCode},
};
use std::io::{stdout, Write};

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
            Print("    Welcome to OxidizeML v0.2 Interface\n"),
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
                    show_placeholder(&mut stdout, menu_items[selected_index].trim())?;
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
