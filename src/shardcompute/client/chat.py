#!/usr/bin/env python3
"""Simple CLI chat client for ShardCompute."""

import argparse
import asyncio
import sys
from typing import Optional, List
import aiohttp
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False


class ChatClient:
    """
    Interactive chat client for ShardCompute cluster.
    
    Connects to coordinator and sends inference requests.
    """
    
    def __init__(
        self,
        coordinator_url: str,
        tokenizer_path: Optional[str] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_tokens: Optional[List[int]] = None,
    ):
        """
        Initialize ChatClient.
        
        Args:
            coordinator_url: URL of coordinator server
            tokenizer_path: Path to tokenizer (optional)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            stop_tokens: Token IDs that stop generation (default: [2] for EOS)
        """
        self.coordinator_url = coordinator_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop_tokens = stop_tokens if stop_tokens is not None else [2]  # Default EOS for Llama
        
        # Load tokenizer if available
        self.tokenizer = None
        if tokenizer_path and HAS_TOKENIZER:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                # Try to get EOS token from tokenizer
                if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                    self.stop_tokens = [self.tokenizer.eos_token_id]
            except Exception as e:
                print(f"Warning: Could not load tokenizer: {e}")
        
        self.console = Console()
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def connect(self) -> bool:
        """Connect to coordinator and check status."""
        self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.get(
                f"{self.coordinator_url}/api/status"
            ) as response:
                if response.status != 200:
                    self.console.print("[red]Cannot connect to coordinator[/red]")
                    return False
                
                status = await response.json()
                
                if not status.get("cluster_ready"):
                    self.console.print("[yellow]Cluster not ready[/yellow]")
                    return False
                
                workers = status.get("workers", 0)
                self.console.print(f"[green]Connected to cluster ({workers} workers)[/green]")
                return True
                
        except Exception as e:
            self.console.print(f"[red]Connection error: {e}[/red]")
            return False
    
    async def disconnect(self):
        """Disconnect from coordinator."""
        if self.session:
            await self.session.close()
    
    async def generate(self, prompt: str) -> Optional[str]:
        """
        Generate response for a prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            Generated response or None on error
        """
        # Tokenize
        if self.tokenizer:
            input_ids = self.tokenizer.encode(prompt)
        else:
            # Simple fallback: use character codes
            input_ids = [ord(c) % 32000 for c in prompt]
        
        # Store input length to strip from output
        input_length = len(input_ids)
        
        # Send request
        payload = {
            "input_ids": input_ids,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop_tokens": self.stop_tokens,
        }
        
        try:
            async with self.session.post(
                f"{self.coordinator_url}/api/inference",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    self.console.print(f"[red]Error: {error}[/red]")
                    return None
                
                data = await response.json()
                
                # Decode response - strip input tokens first!
                output_ids = data.get("output_ids", [])
                
                # The model returns input_ids + generated_ids, so strip the input
                generated_ids = output_ids[input_length:] if len(output_ids) > input_length else output_ids
                
                if self.tokenizer:
                    response_text = self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                    )
                else:
                    response_text = "".join(chr(min(i, 127)) for i in generated_ids)
                
                # Print timing if available
                timing = data.get("timing", {})
                if timing:
                    total_ms = timing.get("total_ms", 0)
                    tokens_per_sec = timing.get("tokens_per_second", 0)
                    num_generated = len(generated_ids)
                    self.console.print(
                        f"[dim]({total_ms:.0f}ms, {tokens_per_sec:.1f} tok/s, {num_generated} tokens)[/dim]"
                    )
                
                return response_text
                
        except asyncio.TimeoutError:
            self.console.print("[red]Request timed out[/red]")
            return None
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            return None
    
    async def chat_loop(self):
        """Run interactive chat loop."""
        self.console.print(Panel(
            "ShardCompute Chat Client\n"
            "Type 'quit' or 'exit' to end the session.\n"
            "Type 'clear' to clear the screen.",
            title="Welcome",
        ))
        
        while True:
            try:
                # Get user input
                prompt = Prompt.ask("\n[bold blue]You[/bold blue]")
                
                if not prompt:
                    continue
                
                if prompt.lower() in ["quit", "exit"]:
                    self.console.print("[dim]Goodbye![/dim]")
                    break
                
                if prompt.lower() == "clear":
                    self.console.clear()
                    continue
                
                # Generate response
                self.console.print("[bold green]Assistant[/bold green]: ", end="")
                
                response = await self.generate(prompt)
                
                if response:
                    self.console.print(response)
                
            except KeyboardInterrupt:
                self.console.print("\n[dim]Interrupted[/dim]")
                break
            except EOFError:
                break


async def main_async(args):
    """Async main function."""
    client = ChatClient(
        coordinator_url=args.coordinator_url,
        tokenizer_path=args.tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    try:
        if not await client.connect():
            return 1
        
        await client.chat_loop()
        return 0
        
    finally:
        await client.disconnect()


def main():
    """Entry point for chat client."""
    parser = argparse.ArgumentParser(
        description="ShardCompute Chat Client"
    )
    parser.add_argument(
        "--coordinator-url",
        type=str,
        default="http://localhost:8080",
        help="Coordinator URL",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold",
    )
    
    args = parser.parse_args()
    
    exit_code = asyncio.run(main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
