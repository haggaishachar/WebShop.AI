"""
Configurable runner for WebShop environments.

This script creates a runner that accepts:
- observation_mode: 'text' (default) or 'html'
- num_products: Number of products to use (default: DEBUG_PROD_SIZE)
- policy: Policy class to use (default: RandomPolicy)

Usage as CLI:
    python run_web_agent_env.py --observation-mode text --num-products 100 --policy random

Usage as Python API:
    from run_envs.run_web_agent_env import create_env, run_episode
    from web_agent_site.models import RandomPolicy
    
    env = create_env(observation_mode='text', num_products=100)
    policy = RandomPolicy()
    stats = run_episode(env, policy)
"""
import argparse
import sys
from os.path import dirname, abspath, join

# Add parent directory to path to enable imports
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import gymnasium as gym
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markup import escape
from rich.syntax import Syntax
import json

from web_agent_site.envs import WebAgentTextEnv, WebAgentSiteEnv
from web_agent_site.models import RandomPolicy, HumanPolicy, PaperRulePolicy, SimpleRulePolicy
from web_agent_site.utils import DEBUG_PROD_SIZE


def create_env(observation_mode='text', num_products=DEBUG_PROD_SIZE, use_site_env=False):
    """
    Create and return a WebShop environment.
    
    Args:
        observation_mode (str): Observation mode ('text' or 'html')
        num_products (int or None): Number of products to use
        use_site_env (bool): If True, use WebAgentSiteEnv; otherwise use WebAgentTextEnv
        
    Returns:
        gym.Env: Configured environment
    """
    if use_site_env:
        try:
            env = WebAgentSiteEnv(
                observation_mode=observation_mode,
                render=False,
                num_products=num_products
            )
        except FileNotFoundError as e:
            if 'chromedriver' in str(e).lower():
                print("\n❌ ChromeDriver not found!")
                print("\nTo run the site environment, you need to:")
                print("  1. Download ChromeDriver for your system from:")
                print("     https://chromedriver.chromium.org/downloads")
                print("  2. Or use: sudo apt-get install chromium-chromedriver (on Ubuntu/Debian)")
                print("  3. Place the binary at: web_agent_site/envs/chromedriver")
                print("     Or ensure 'chromedriver' is in your PATH")
                exit(1)
            raise
        except OSError as e:
            if 'Exec format error' in str(e):
                print("\n❌ ChromeDriver architecture mismatch!")
                print("\nThe chromedriver binary is not compatible with your system.")
                print("Please download the correct version for your OS and architecture:")
                print("  - Linux: https://chromedriver.chromium.org/downloads")
                print("  - Or use: sudo apt-get install chromium-chromedriver (on Ubuntu/Debian)")
                exit(1)
            raise
        except Exception as e:
            if 'ERR_CONNECTION_REFUSED' in str(e) or 'Connection refused' in str(e):
                print("\n❌ Cannot connect to WebShop server!")
                print("\nThe site environment requires the Flask app to be running.")
                print("\nTo fix this:")
                print("  1. In one terminal, start the Flask app:")
                print("     make run-dev")
                print("\n  2. In another terminal, run this script again")
                exit(1)
            raise
    else:
        env = gym.make(
            'WebAgentTextEnv-v0',
            observation_mode=observation_mode,
            num_products=num_products
        )
    
    return env


def format_observation(observation):
    """Format observation text for better readability."""
    # Split by [SEP] and clean up
    parts = observation.split('[SEP]')
    formatted = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Format different types of content
        if part.startswith('Instruction:'):
            formatted.append(f"[bold cyan]{part}[/bold cyan]")
        elif part.startswith('WebShop'):
            formatted.append(f"[bold yellow]{part}[/bold yellow]")
        elif part.startswith('Price:'):
            formatted.append(f"[green]{part}[/green]")
        elif part.startswith('Rating:'):
            formatted.append(f"[yellow]{part}[/yellow]")
        elif part.startswith('Page '):
            formatted.append(f"[magenta]{part}[/magenta]")
        elif part in ['Search', 'Buy Now', 'Description', 'Features', 'Reviews']:
            formatted.append(f"[bold]{part}[/bold]")
        elif part.startswith('B0'):  # Product IDs
            formatted.append(f"[blue]{part}[/blue]")
        else:
            formatted.append(part)
    
    return '\n'.join(formatted)


def format_actions(available_actions):
    """Format available actions for better readability."""
    if not available_actions:
        return "No actions available"
    
    lines = []
    if available_actions.get('has_search_bar'):
        lines.append("[green]✓[/green] Search bar available")
    else:
        lines.append("[red]✗[/red] No search bar")
    
    clickables = available_actions.get('clickables', [])
    if clickables:
        lines.append(f"\n[bold]Clickable options ({len(clickables)}):[/bold]")
        for i, action in enumerate(clickables, 1):
            lines.append(f"  {i:2d}. [cyan]{action}[/cyan]")
    
    return '\n'.join(lines)


def run_episode(env, policy, episode_num=1, max_steps=100):
    """
    Run a single episode using the given environment and policy.
    
    Args:
        env (gym.Env): Environment to run
        policy: Policy object with a forward method
        episode_num (int): Episode number for display
        max_steps (int): Maximum number of steps per episode (default: 100)
        
    Returns:
        dict: Episode statistics (total_reward, steps, reward_components)
    """
    console = Console()
    observation, info = env.reset()
    total_reward = 0.0
    steps = 0
    reward_components = None
    
    # Reset policy state if it has a reset method
    if hasattr(policy, 'reset'):
        policy.reset()
    
    while True:
        steps += 1
        
        # Check if we've exceeded max steps
        if steps > max_steps:
            console.print(f"\n[bold yellow]⚠ Episode truncated after {max_steps} steps[/bold yellow]")
            break
        
        # Display step header with episode number
        console.print(f"\n[bold white on blue] EPISODE {episode_num} | STEP {steps} [/bold white on blue]")
        
        # Display observation in a panel
        formatted_obs = format_observation(observation)
        console.print(Panel(
            formatted_obs,
            title="[bold]🔍 Observation[/bold]",
            border_style="blue",
            padding=(1, 2)
        ))
        
        # Display available actions
        available_actions = env.unwrapped.get_available_actions()
        formatted_actions = format_actions(available_actions)
        console.print(Panel(
            formatted_actions,
            title="[bold]⚡ Available Actions[/bold]",
            border_style="green",
            padding=(1, 2)
        ))
        
        # Get and display action
        action = policy.forward(observation, available_actions)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        
        # Display action taken and reward
        reward_color = "green" if reward > 0 else "yellow" if reward == 0 else "red"
        console.print(f"\n[bold]➤ Action:[/bold] [cyan]{escape(action)}[/cyan]")
        console.print(f"[bold]➤ Reward:[/bold] [{reward_color}]{reward}[/{reward_color}]")
        console.print(f"[bold]➤ Cumulative Reward:[/bold] {total_reward}")
        
        if done:
            console.print("\n[bold green]✓ Episode completed![/bold green]")
            # Extract reward components from the environment
            try:
                session_id = env.unwrapped.browser.session_id
                if hasattr(env.unwrapped, 'server') and session_id in env.unwrapped.server.user_sessions:
                    reward_components = env.unwrapped.server.user_sessions[session_id].get('verbose_info', {})
            except (AttributeError, KeyError):
                pass
            break
        
        # Add separator between steps
        console.print("\n" + "─" * 80)
    
    return {
        'total_reward': total_reward,
        'steps': steps,
        'reward_components': reward_components
    }


def main():
    """Main entry point for the runner."""
    parser = argparse.ArgumentParser(
        description='Run WebShop environment with configurable parameters'
    )
    parser.add_argument(
        '--observation-mode',
        type=str,
        default='text',
        choices=['text', 'html', 'text_rich'],
        help='Observation mode (default: text)'
    )
    parser.add_argument(
        '--num-products',
        type=int,
        default=DEBUG_PROD_SIZE,
        help=f'Number of products to use (default: {DEBUG_PROD_SIZE})'
    )
    parser.add_argument(
        '--policy',
        type=str,
        default='random',
        choices=['random', 'human', 'paper_rule', 'simple_rule'],
        help='Policy to use (default: random)'
    )
    parser.add_argument(
        '--use-site-env',
        action='store_true',
        help='Use WebAgentSiteEnv instead of WebAgentTextEnv (requires ChromeDriver and running server)'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=1,
        help='Number of episodes to run (default: 1)'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=100,
        help='Maximum steps per episode (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Print run parameters at the beginning
    print(f"\n🚀 Starting WebShop environment")
    print(f"   Observation mode: {args.observation_mode}")
    print(f"   Number of products: {args.num_products}")
    print(f"   Policy: {args.policy}")
    print(f"   Environment: {'WebAgentSiteEnv' if args.use_site_env else 'WebAgentTextEnv'}")
    print(f"   Number of episodes: {args.num_episodes}")
    print(f"   Max steps per episode: {args.max_steps}\n")
    
    # Create policy
    if args.policy == 'random':
        policy = RandomPolicy()
    elif args.policy == 'human':
        policy = HumanPolicy()
    elif args.policy == 'paper_rule':
        policy = PaperRulePolicy()
    elif args.policy == 'simple_rule':
        policy = SimpleRulePolicy()
    else:
        raise ValueError(f'Unknown policy: {args.policy}')
    
    # Create environment
    env = create_env(
        observation_mode=args.observation_mode,
        num_products=args.num_products,
        use_site_env=args.use_site_env
    )
    
    try:
        console = Console()
        all_stats = []
        
        for episode in range(1, args.num_episodes + 1):
            if args.num_episodes > 1:
                console.print(f"\n[bold magenta]{'=' * 80}[/bold magenta]")
                console.print(f"[bold magenta]Starting Episode {episode} of {args.num_episodes}[/bold magenta]")
                console.print(f"[bold magenta]{'=' * 80}[/bold magenta]")
            
            stats = run_episode(env, policy, episode_num=episode, max_steps=args.max_steps)
            all_stats.append(stats)
            
            # Display episode statistics with reward components
            summary_lines = [
                f"[bold green]Episode {episode} Completed Successfully![/bold green]",
                "",
                f"[bold]Total Reward:[/bold]  {stats['total_reward']:.4f}",
                f"[bold]Total Steps:[/bold]   {stats['steps']}",
                f"[bold]Avg Reward:[/bold]    {stats['total_reward'] / stats['steps']:.4f}",
            ]
            
            # Add reward components breakdown if available
            if stats.get('reward_components'):
                rc = stats['reward_components']
                summary_lines.extend([
                    "",
                    "[bold cyan]Reward Components:[/bold cyan]",
                    f"  [bold]Type Match (r_type):[/bold]      {rc.get('r_type', 0.0):.4f}",
                    f"  [bold]Attributes (r_att):[/bold]       {rc.get('r_att', 0.0):.4f}",
                    f"  [bold]Options (r_option):[/bold]       {rc.get('r_option', 0.0):.4f}" if 'r_option' in rc else "  [bold]Options (r_option):[/bold]       N/A",
                    f"  [bold]Price Match (r_price):[/bold]    {rc.get('r_price', 0.0):.4f}" if 'r_price' in rc else "  [bold]Price Match (r_price):[/bold]    N/A",
                ])
            
            summary = '\n'.join(summary_lines)
            console.print(Panel(
                summary,
                title=f"[bold]📊 Episode {episode} Summary[/bold]",
                border_style="green",
                padding=(1, 2)
            ))
        
        # Display overall statistics if multiple episodes
        if args.num_episodes > 1:
            total_reward = sum(s['total_reward'] for s in all_stats)
            total_steps = sum(s['steps'] for s in all_stats)
            avg_reward_per_episode = total_reward / args.num_episodes
            avg_steps_per_episode = total_steps / args.num_episodes
            
            # Calculate success metrics
            successful_episodes = sum(1 for s in all_stats if s['total_reward'] >= 1.0)
            success_rate = (successful_episodes / args.num_episodes) * 100
            
            # Create detailed table with all episodes and their reward components
            episodes_with_components = [s for s in all_stats if s.get('reward_components')]
            if episodes_with_components:
                table = Table(
                    title="📊 Episode Reward Components Breakdown\n[dim]Reward = (# matched attrs + # matched options + price_ok) / (# total attrs + # total options + 1) × Type[/dim]",
                    show_header=True,
                    header_style="bold cyan"
                )
                table.add_column("Ep", style="cyan", justify="center")
                table.add_column("Steps", style="dim", justify="right")
                table.add_column("Total", style="dim", justify="right")
                table.add_column("Reward", style="green", justify="right")
                table.add_column("Type", justify="right")
                table.add_column("Attrs", justify="right")
                table.add_column("Options", justify="right")
                table.add_column("Price", justify="right")
                
                for idx, stats in enumerate(all_stats, start=1):
                    rc = stats.get('reward_components')
                    if rc:
                        # Calculate total from weights (total = 1 / w_price)
                        total_components = int(round(1 / rc.get('w_price', 1))) if 'w_price' in rc else 0
                        
                        # Color code the total reward and r_type
                        total_reward_str = f"{stats['total_reward']:.4f}"
                        r_type_val = rc.get('r_type', 0.0)
                        r_type_str = f"[red]{r_type_val:.4f}[/red]" if r_type_val == 0 else f"{r_type_val:.4f}"
                        
                        # Convert boolean r_price to float for display
                        r_price_val = rc.get('r_price', False)
                        r_price_str = f"{float(r_price_val):.4f}" if 'r_price' in rc else "N/A"
                        
                        table.add_row(
                            str(idx),
                            str(stats['steps']),
                            str(total_components) if total_components > 0 else "N/A",
                            total_reward_str,
                            r_type_str,
                            f"{rc.get('r_att', 0.0):.4f}",
                            f"{rc.get('r_option', 0.0):.4f}" if 'r_option' in rc else "N/A",
                            r_price_str,
                        )
                    else:
                        table.add_row(
                            str(idx),
                            str(stats['steps']),
                            "N/A",
                            f"{stats['total_reward']:.4f}",
                            "N/A",
                            "N/A",
                            "N/A",
                            "N/A",
                        )
                
                console.print(table)
            
            # Build and display overall summary after the table
            overall_summary_lines = [
                f"[bold cyan]All Episodes Completed![/bold cyan]",
                "",
                f"[bold]Total Episodes:[/bold]        {args.num_episodes}",
                f"[bold]Successful Episodes:[/bold]   {successful_episodes}",
                f"[bold]Success Rate:[/bold]          {success_rate:.2f}%",
                f"[bold]Average Reward:[/bold]        {avg_reward_per_episode:.4f}",
                f"[bold]Average Steps:[/bold]         {avg_steps_per_episode:.2f}",
            ]
            
            overall_summary = '\n'.join(overall_summary_lines)
            console.print(Panel(
                overall_summary,
                title="[bold]📈 Overall Summary[/bold]",
                border_style="cyan",
                padding=(1, 2)
            ))
        
    finally:
        env.close()


if __name__ == '__main__':
    main()

