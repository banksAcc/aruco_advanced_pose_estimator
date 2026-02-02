
"""BLE communication utilities for the capture application.

This module discovers and connects to a BLE device, relaying start and stop
commands between the hardware and the session manager. It also allows sending
arbitrary messages to the device through a queue.
"""

from __future__ import annotations

import asyncio
from typing import Optional, TYPE_CHECKING

from bleak import BleakClient, BleakScanner

from utils.config_models import AppConfig
from utils.logger import get_logger
from utils.utils import Encode_as_bytes

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from session_manager import SessionManager

START_CMD = "START" 
END_CMD = "END"  # user confirmed END

log = get_logger("BLE")

async def _on_notify(sender: int | object, data: bytearray, session_mgr: "SessionManager") -> None:
    """
    Handle notifications from the BLE device.
    Nota: 'sender' in Bleak moderni è un oggetto Characteristic, non un int, 
    ma qui usiamo un tipo generico per sicurezza.
    """
    msg = data.decode(errors="ignore").strip().upper()
    
    if msg == START_CMD:
        await session_mgr.handle_start_command()
    elif msg in (END_CMD, "STOP"): # Gestione unificata stop
        await session_mgr.handle_end_command()
    else:
        log.warning(f"Unknown message: {msg!r}")

async def _discover_address(cfg: AppConfig) -> str | None:
    """Find the BLE device address from the configuration or by scanning."""

    addr = cfg.ble.addr
    if addr:
        return addr
    name_prefix = cfg.ble.name_prefix
    timeout = float(cfg.ble.scan_timeout)
    log.info(f"Scanning {timeout:.1f}s for '{name_prefix}*' ...")
    devices = await BleakScanner.discover(timeout=timeout)
    for d in devices:
        if d.name and d.name.startswith(name_prefix):
            log.info(f"Found: {d.name} @ {d.address}")
            return d.address
    log.warning("No device found.")
    return None


async def run_ble_client(
    cfg: AppConfig,
    session_mgr: "SessionManager",
    out_queue: asyncio.Queue[Optional[Encode_as_bytes]],
) -> None:
    """Connect to the ESP32 and bridge BLE messages to the session manager."""

    address = await _discover_address(cfg)
    if not address:
        log.error("No address: exiting.")
        return

    # UUIDs definiti qui per ora (da spostare in config come da tuo TODO)
    NUS_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
    NUS_RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"

    while True:
        try:
            log.info(f"Connecting to {address} ...")
            async with BleakClient(address, timeout=10.0) as client:
                log.info(f"Connected: {client.is_connected}")
                if not client.is_connected:
                    raise RuntimeError("Connection failed")

                # --- CORREZIONE QUI SOTTO ---
                
                # Definiamo un wrapper che rispetta la firma richiesta da Bleak (sender, data)
                # ma che al suo interno ha accesso a 'session_mgr'
                async def notification_handler(sender, data):
                    await _on_notify(sender, data, session_mgr)

                await session_mgr.on_ble_connected()
                
                # Passiamo il wrapper a start_notify
                await client.start_notify(NUS_TX_UUID, notification_handler)
                
                # -----------------------------

                sender_task = None
                try:
                    async def send_queued() -> None:
                        """Send messages from the queue to the BLE device."""
                        while True:
                            msg = await out_queue.get()
                            if msg is None:
                                break
                            try:
                                await client.write_gatt_char(
                                    NUS_RX_UUID,
                                    msg.as_bytes(),
                                )
                            except Exception as e:
                                log.error(f"send error: {e}")

                    sender_task = asyncio.create_task(send_queued())

                    # Loop principale di mantenimento connessione
                    while client.is_connected:
                        await asyncio.sleep(0.5)

                finally:
                    if sender_task is not None:
                        sender_task.cancel()
                        try:
                            await sender_task
                        except asyncio.CancelledError:
                            pass
                    log.info("Disconnected.")
                    await session_mgr.on_ble_disconnected()

        except KeyboardInterrupt:
            log.info("Interrupted by user.")
            await session_mgr.on_ble_disconnected()
            return
        except Exception as e:
            log.error(f"Error: {e}")
            await session_mgr.on_ble_disconnected()

        log.info("Retrying in 2s ...")
        await asyncio.sleep(2.0)
