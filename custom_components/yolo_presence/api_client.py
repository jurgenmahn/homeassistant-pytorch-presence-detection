"""API client for communicating with the YOLO Presence Processing Server."""
import asyncio
import json
import logging
import socket
import time
from typing import Dict, Any, Optional, List, Callable, Tuple
import aiohttp
from aiohttp.client_exceptions import ClientConnectorError, ClientError, ClientPayloadError
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
import homeassistant.util.dt as dt_util
from .const import ATTR_DEVICE_ID, CONF_PROCESSING_SERVER_PORT

_LOGGER = logging.getLogger(__name__)

class YoloProcessingApiClient:
    """API client for the YOLO Presence Processing Server."""

    def __init__(
        self,
        hass: HomeAssistant,
        server_url: str = None,
        host: str = None,
        port: int = 5000,
        detector_id: str = None,
        update_interval: int = 30,
        server_port: int = None,
        use_tcp: bool = True,
    ) -> None:
        """Initialize the API client."""
        self.hass = hass
        
        # Handle backward compatibility and different initialization methods
        # For TCP connections, we expect server_url to be just the hostname, not a URL
        if server_url:
            # Check if server_url includes a protocol (http://)
            if server_url.startswith("http://") or server_url.startswith("https://"):
                # Extract host from server URL
                from urllib.parse import urlparse
                parsed_url = urlparse(server_url)
                self.host = parsed_url.netloc.split(':')[0] or parsed_url.path
                _LOGGER.warning(
                    "HTTP URL format detected for server_url (%s). Using host: %s with TCP", 
                    server_url, self.host
                )
            else:
                # Use as-is for direct TCP connection
                self.host = server_url
        else:
            self.host = host
            
        # Port priority: server_port param > port param > default
        self.port = server_port or port
        self.detector_id = detector_id
        
        _LOGGER.info("Initializing YOLO TCP client for %s:%s", self.host, self.port)
        self._socket = None
        self._reader = None
        self._writer = None
        self._connected = False
        self._connection_lock = asyncio.Lock()
        self._connection_retry_count = 0
        self._max_connection_retries = 0  # 0 means infinite retries - never give up
        self._connection_retry_delay = 5  # seconds
        self._connection_timeout = 10  # seconds
        self._heartbeat_interval = 15  # seconds - reduced for more frequent checks
        self._last_heartbeat_sent = 0
        self._last_heartbeat_received = 0
        self._heartbeat_response_timeout = 30  # seconds - how long to wait for response
        self._heartbeat_task = None
        self._tcp_connection_task = None
        self._update_callbacks = []
        self._human_detected = False
        self._pet_detected = False
        self._human_count = 0
        self._pet_count = 0
        self._last_update = None
        self._stop_event = asyncio.Event()
        self._reconnect_event = asyncio.Event()
        self._health_check_interval = 30  # seconds - reduced for more frequent checks
        self._health_check_task = None
        self._last_successful_connection = 0
        self._backoff_factor = 1.2  # Lower backoff factor for more gradual increases
        self._max_backoff = 300  # 5 minutes
        
        # Connection state
        self._connection_error_reported = False  # Track if we've already shown an error
        self._heartbeat_missed_count = 0
        self._max_missed_heartbeats = 2  # After this many missed heartbeats, force reconnect

    async def async_connect(self) -> bool:
        """Connect to the processing server."""
        if self._connected:
            return True

        async with self._connection_lock:
            if self._connected:  # Check again inside lock
                return True

            try:
                _LOGGER.debug(
                    "Connecting to YOLO Processing Server at %s:%s",
                    self.host,
                    self.port,
                )

                # Create TCP connection with timeout
                try:
                    connection_future = asyncio.open_connection(
                        self.host, self.port
                    )
                    self._reader, self._writer = await asyncio.wait_for(
                        connection_future, timeout=self._connection_timeout
                    )
                except (asyncio.TimeoutError, ConnectionRefusedError) as err:
                    _LOGGER.error(
                        "Failed to connect to YOLO Processing Server: %s", err
                    )
                    await self._handle_connection_failure()
                    return False

                # Send authentication message
                auth_message = {
                    "type": "auth",
                    "detector_id": self.detector_id,
                }
                if not await self._send_message(auth_message):
                    _LOGGER.error("Failed to send authentication message")
                    await self._close_connection()
                    await self._handle_connection_failure()
                    return False

                # Wait for auth response
                try:
                    response = await asyncio.wait_for(
                        self._read_message(), timeout=self._connection_timeout
                    )
                except asyncio.TimeoutError:
                    _LOGGER.error("Authentication response timeout")
                    await self._close_connection()
                    await self._handle_connection_failure()
                    return False

                if response.get("type") != "auth_success":
                    _LOGGER.error(
                        "Authentication failed: %s", response.get("message", "Unknown error")
                    )
                    await self._close_connection()
                    await self._handle_connection_failure()
                    return False

                self._connected = True
                self._connection_retry_count = 0
                self._last_successful_connection = time.time()
                self._connection_retry_delay = 5  # Reset backoff
                self._connection_error_reported = False  # Reset error flag
                self._heartbeat_missed_count = 0  # Reset missed heartbeat counter
                self._last_heartbeat_received = time.time()  # Initialize heartbeat timestamp
                
                # Start heartbeat and connection monitoring
                self._start_heartbeat()
                self._start_health_check()
                
                # Start TCP connection loop
                self._tcp_connection_task = asyncio.create_task(
                    self._tcp_connection_loop()
                )
                
                # Fire connection event
                await self._fire_connection_event("connected")
                
                _LOGGER.info(
                    "Connection established with heartbeat interval=%ss, "
                    "health check interval=%ss",
                    self._heartbeat_interval,
                    self._health_check_interval
                )
                
                _LOGGER.info(
                    "Successfully connected to YOLO Processing Server at %s:%s",
                    self.host,
                    self.port,
                )
                return True

            except Exception as err:
                _LOGGER.exception(
                    "Unexpected error connecting to YOLO Processing Server: %s", err
                )
                await self._close_connection()
                await self._handle_connection_failure()
                return False

    async def _handle_connection_failure(self) -> None:
        """Handle connection failure with progressive backoff."""
        self._connection_retry_count += 1
        
        try:
            # Calculate backoff with exponential factor, with safety limits
            # Limit the exponent to prevent overflow
            exponent = min(self._connection_retry_count - 1, 20)  # Limit to prevent overflow
            
            # Calculate backoff with capped exponent
            backoff = min(
                self._connection_retry_delay * (self._backoff_factor ** exponent),
                self._max_backoff
            )
            
            # Ensure backoff is within reasonable limits
            backoff = max(self._connection_retry_delay, min(backoff, self._max_backoff))
            
        except (OverflowError, ValueError):
            # Fallback in case of any calculation errors
            _LOGGER.warning("Backoff calculation error, using maximum backoff")
            backoff = self._max_backoff
        
        _LOGGER.warning(
            "Connection to YOLO Processing Server failed. "
            "Retry %s/%s in %.1f seconds",
            self._connection_retry_count,
            self._max_connection_retries if self._max_connection_retries > 0 else "∞",
            backoff,
        )
        
        # Fire connection event
        await self._fire_connection_event("disconnected")
        
        # Schedule reconnection
        self._reconnect_event.set()

    async def _close_connection(self) -> None:
        """Close the connection to the processing server."""
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception as err:
                _LOGGER.debug("Error closing connection: %s", err)
        
        self._reader = None
        self._writer = None
        self._connected = False

    def _start_heartbeat(self) -> None:
        """Start the heartbeat task."""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            _LOGGER.debug("Heartbeat monitoring started")

    def _start_health_check(self) -> None:
        """Start the health check task."""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            _LOGGER.debug("Health check monitoring started")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to keep the connection alive and monitor responses."""
        try:
            while not self._stop_event.is_set():
                if self._connected:
                    current_time = time.time()
                    
                    # Check if it's time to send a heartbeat
                    if current_time - self._last_heartbeat_sent >= self._heartbeat_interval:
                        # Send heartbeat to check if connection is still alive
                        _LOGGER.debug("Sending heartbeat to check connection")
                        if await self._send_message({"type": "heartbeat"}):
                            self._last_heartbeat_sent = current_time
                        else:
                            _LOGGER.warning("Failed to send heartbeat, will attempt to reconnect")
                            # Trigger reconnection
                            self._reconnect_event.set()
                            await asyncio.sleep(1)  # Small delay before continuing
                            continue
                    
                    # Check if we've received a heartbeat response recently
                    time_since_last_received = current_time - self._last_heartbeat_received
                    if time_since_last_received > self._heartbeat_response_timeout:
                        self._heartbeat_missed_count += 1
                        _LOGGER.warning(
                            "No heartbeat response received in %s seconds (count: %s/%s)",
                            time_since_last_received,
                            self._heartbeat_missed_count,
                            self._max_missed_heartbeats
                        )
                        
                        # If we've missed too many heartbeats, reconnect
                        if self._heartbeat_missed_count >= self._max_missed_heartbeats:
                            _LOGGER.warning(
                                "Missed %s heartbeats, forcing reconnection",
                                self._heartbeat_missed_count
                            )
                            await self._close_connection()
                            self._reconnect_event.set()
                            # Reset counter
                            self._heartbeat_missed_count = 0
                            await asyncio.sleep(1)  # Small delay before continuing
                            continue
                
                await asyncio.sleep(2)  # Check more frequently
        except asyncio.CancelledError:
            _LOGGER.debug("Heartbeat task cancelled")
        except Exception as err:
            _LOGGER.exception("Unexpected error in heartbeat loop: %s", err)
            # Force reconnection on unexpected errors
            if not self._stop_event.is_set() and self._connected:
                self._reconnect_event.set()

    async def _health_check_loop(self) -> None:
        """
        Monitor connection health and reconnect if needed.
        This is a more aggressive health check that ensures the connection
        is properly maintained and will force reconnection if needed.
        """
        try:
            while not self._stop_event.is_set():
                try:
                    # Check if we're currently connected
                    if self._connected:
                        # We're connected, just make sure the connection is healthy
                        current_time = time.time()
                        
                        # The heartbeat loop handles most active connection checking,
                        # but we'll do a redundant check here for extra reliability
                        time_since_last_received = current_time - self._last_heartbeat_received
                        if time_since_last_received > self._heartbeat_response_timeout * 1.5:
                            _LOGGER.warning(
                                "Health check detected no server activity for %s seconds, "
                                "forcing reconnection",
                                time_since_last_received
                            )
                            await self._close_connection()
                            self._reconnect_event.set()
                    else:
                        # Not connected - make sure reconnection event is set
                        if not self._reconnect_event.is_set():
                            _LOGGER.info("Health check detected disconnected state, triggering reconnection")
                            self._reconnect_event.set()
                    
                    # Wait for reconnect event or timeout
                    try:
                        await asyncio.wait_for(
                            self._reconnect_event.wait(), 
                            timeout=self._health_check_interval
                        )
                        # Clear the event before attempting to reconnect
                        self._reconnect_event.clear()
                        
                        # Don't reconnect if we're stopping
                        if not self._stop_event.is_set():
                            # We're going to try connecting again, so reset any error flags
                            self._connection_error_reported = False
                            
                            # Log reconnection attempt
                            _LOGGER.info(
                                "Attempting to reconnect to %s:%s (attempt %s)",
                                self.host, self.port, self._connection_retry_count + 1
                            )
                            
                            # Try to connect
                            await self.async_connect()
                    except asyncio.TimeoutError:
                        # Normal timeout, continue checking
                        pass
                    
                except asyncio.CancelledError:
                    raise  # Re-raise CancelledError to properly handle it
                except Exception as health_err:
                    # Log error but continue the health check loop
                    _LOGGER.error("Error in health check cycle: %s", health_err)
                    # Force reconnection on unexpected errors
                    if not self._stop_event.is_set() and self._connected:
                        await self._close_connection()
                        self._reconnect_event.set()
                    await asyncio.sleep(5)  # Short sleep on error
                
        except asyncio.CancelledError:
            _LOGGER.debug("Health check task cancelled")
        except Exception as err:
            _LOGGER.exception("Unexpected error in health check loop: %s", err)
            # Make sure we try to reconnect even if there's an unexpected error
            if not self._stop_event.is_set():
                self._reconnect_event.set()

    async def _tcp_connection_loop(self) -> None:
        """Handle the TCP connection and incoming messages."""
        try:
            while not self._stop_event.is_set() and self._connected:
                try:
                    message = await self._read_message()
                    if message:
                        await self._handle_message(message)
                except (ConnectionResetError, BrokenPipeError) as conn_err:
                    # Handle specific connection errors more gracefully
                    if not self._stop_event.is_set():
                        _LOGGER.warning("Connection broken: %s. Will attempt to reconnect.", conn_err)
                        await self._close_connection()
                        self._reconnect_event.set()
                        break
                except asyncio.IncompleteReadError:
                    # Handle incomplete read errors which indicate connection issues
                    if not self._stop_event.is_set():
                        _LOGGER.warning("Connection interrupted (incomplete read). Will attempt to reconnect.")
                        await self._close_connection()
                        self._reconnect_event.set()
                        break
                except asyncio.CancelledError:
                    raise
                except Exception as err:
                    if not self._stop_event.is_set():
                        _LOGGER.error("Error in TCP connection loop: %s", err)
                        await self._close_connection()
                        self._reconnect_event.set()
                        break
        except asyncio.CancelledError:
            _LOGGER.debug("TCP connection loop cancelled")
        except Exception as err:
            _LOGGER.exception("Unexpected error in TCP connection loop: %s", err)
            # Force reconnection on unexpected errors
            if not self._stop_event.is_set():
                self._connected = False
                self._reconnect_event.set()

    async def _read_message(self) -> Dict[str, Any]:
        """Read a message from the server."""
        if not self._reader:
            raise ConnectionError("Not connected to server")
        
        try:
            # Read message length (4 bytes)
            length_bytes = await self._reader.readexactly(4)
            length = int.from_bytes(length_bytes, byteorder="big")
            
            # Read message data
            data = await self._reader.readexactly(length)
            message = json.loads(data.decode("utf-8"))
            
            # Log received message but don't handle it here
            # Heartbeat handling is now done in _handle_message
            message_type = message.get("type", "unknown")
            _LOGGER.debug("Received message type: %s", message_type)
                
            return message
        except asyncio.IncompleteReadError:
            _LOGGER.warning("Connection closed by server")
            raise ConnectionError("Connection closed by server")
        except json.JSONDecodeError as err:
            _LOGGER.error("Failed to decode message: %s", err)
            raise

    async def _send_message(self, message: Dict[str, Any]) -> bool:
        """
        Send a message to the server.
        
        Returns:
            bool: True if the message was sent successfully, False on connection errors
        """
        if not self._writer:
            _LOGGER.error("Failed to send message: not connected to server")
            return False
        
        try:
            # Encode message to JSON
            data = json.dumps(message).encode("utf-8")
            
            # Send message length (4 bytes) followed by data
            length = len(data)
            self._writer.write(length.to_bytes(4, byteorder="big"))
            self._writer.write(data)
            await self._writer.drain()
            
            _LOGGER.debug("Sent message: %s", message)
            return True
            
        except ConnectionResetError as conn_err:
            _LOGGER.warning("Connection reset while sending message: %s", conn_err)
            await self._close_connection()
            self._reconnect_event.set()
            return False
            
        except BrokenPipeError as pipe_err:
            _LOGGER.warning("Broken pipe while sending message: %s", pipe_err)
            await self._close_connection()
            self._reconnect_event.set()
            return False
            
        except Exception as err:
            _LOGGER.error("Failed to send message: %s", err)
            await self._close_connection()
            self._reconnect_event.set()
            return False

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle a message from the server."""
        message_type = message.get("type")
        
        if message_type == "state_update":
            # Any message from the server, including state updates, should reset heartbeat monitoring
            self._last_heartbeat_received = time.time()
            self._heartbeat_missed_count = 0
            await self._handle_state_update(message)
        elif message_type == "heartbeat":
            # Record that we received a heartbeat
            self._last_heartbeat_received = time.time()
            self._heartbeat_missed_count = 0
            _LOGGER.debug("Received heartbeat from server, connection is healthy")
        elif message_type == "error":
            # Even errors indicate the connection is working
            self._last_heartbeat_received = time.time()
            _LOGGER.error("Error from server: %s", message.get("message", "Unknown error"))
        else:
            # Any message received means the connection is alive
            self._last_heartbeat_received = time.time()
            _LOGGER.warning("Unknown message type: %s", message_type)

    async def _handle_state_update(self, message: Dict[str, Any]) -> None:
        """Handle a state update message."""
        state = message.get("state", {})
        
        # Update state
        human_detected = state.get("human_detected", False)
        pet_detected = state.get("pet_detected", False)
        human_count = state.get("human_count", 0)
        pet_count = state.get("pet_count", 0)
        
        # Check if state has changed
        state_changed = (
            human_detected != self._human_detected
            or pet_detected != self._pet_detected
            or human_count != self._human_count
            or pet_count != self._pet_count
        )
        
        if state_changed:
            # Update internal state
            self._human_detected = human_detected
            self._pet_detected = pet_detected
            self._human_count = human_count
            self._pet_count = pet_count
            self._last_update = dt_util.utcnow()
            
            # Fire events
            if human_detected != self._human_detected:
                await self._fire_human_event(human_detected)
            
            if pet_detected != self._pet_detected:
                await self._fire_pet_event(pet_detected)
            
            if human_count != self._human_count:
                await self._fire_human_count_event(human_count)
            
            if pet_count != self._pet_count:
                await self._fire_pet_count_event(pet_count)
            
            # Call update callbacks
            self._call_update_callbacks()
            
            _LOGGER.debug(
                "State updated: human=%s, pet=%s, human_count=%s, pet_count=%s",
                human_detected,
                pet_detected,
                human_count,
                pet_count,
            )

    async def async_update(self) -> None:
        """Update the state by requesting latest data from the server."""
        if not self._connected:
            try:
                # Reset retry count if it's been a long time since the last attempt
                # This allows automatic retries to resume after manual intervention
                current_time = time.time()
                if current_time - self._last_successful_connection > 600:  # 10 minutes
                    # If it's been a while, reset the retry counter to allow reconnection attempts
                    if self._connection_retry_count >= self._max_connection_retries:
                        _LOGGER.info("Resetting connection retry counter to allow new connection attempts")
                        self._connection_retry_count = 0
                        self._connection_error_reported = False
                
                connected = await self.async_connect()
                if not connected:
                    return
            except Exception as err:
                _LOGGER.error("Failed to connect for update: %s", err)
                return
        
        # Request state update
        if not await self._send_message({"type": "get_state"}):
            _LOGGER.error("Failed to request state update, connection may be broken")
            self._reconnect_event.set()
            return

    async def async_close(self) -> None:
        """Close the connection and stop all tasks."""
        self._stop_event.set()
        
        # Cancel all tasks
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            
        if self._tcp_connection_task and not self._tcp_connection_task.done():
            self._tcp_connection_task.cancel()
        
        # Close the connection
        await self._close_connection()
        
        _LOGGER.info("API client closed")

    def register_update_callback(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register a callback for state updates."""
        self._update_callbacks.append(callback)
        
        def remove_callback() -> None:
            """Remove the callback."""
            if callback in self._update_callbacks:
                self._update_callbacks.remove(callback)
        
        return remove_callback

    def _call_update_callbacks(self) -> None:
        """Call all registered update callbacks."""
        for callback in self._update_callbacks:
            try:
                callback()
            except Exception as err:
                _LOGGER.error("Error in update callback: %s", err)

    async def _fire_connection_event(self, status: str) -> None:
        """Fire a connection status changed event."""
        self.hass.bus.async_fire(
            "yolo_presence_connection_status_changed",
            {
                ATTR_DEVICE_ID: self.detector_id,
                "status": status,
            },
        )

    async def _fire_human_event(self, detected: bool) -> None:
        """Fire a human detected event."""
        self.hass.bus.async_fire(
            "yolo_presence_human_detected",
            {
                ATTR_DEVICE_ID: self.detector_id,
                "detected": detected,
            },
        )

    async def _fire_pet_event(self, detected: bool) -> None:
        """Fire a pet detected event."""
        self.hass.bus.async_fire(
            "yolo_presence_pet_detected",
            {
                ATTR_DEVICE_ID: self.detector_id,
                "detected": detected,
            },
        )

    async def _fire_human_count_event(self, count: int) -> None:
        """Fire a human count changed event."""
        self.hass.bus.async_fire(
            "yolo_presence_human_count_changed",
            {
                ATTR_DEVICE_ID: self.detector_id,
                "count": count,
            },
        )

    async def _fire_pet_count_event(self, count: int) -> None:
        """Fire a pet count changed event."""
        self.hass.bus.async_fire(
            "yolo_presence_pet_count_changed",
            {
                ATTR_DEVICE_ID: self.detector_id,
                "count": count,
            },
        )

    @property
    def human_detected(self) -> bool:
        """Return whether a human is detected."""
        return self._human_detected

    @property
    def pet_detected(self) -> bool:
        """Return whether a pet is detected."""
        return self._pet_detected

    @property
    def human_count(self) -> int:
        """Return the number of humans detected."""
        return self._human_count

    @property
    def pet_count(self) -> int:
        """Return the number of pets detected."""
        return self._pet_count

    @property
    def last_update(self) -> Optional[str]:
        """Return the timestamp of the last update."""
        if self._last_update:
            return self._last_update.isoformat()
        return None

    @property
    def is_connected(self) -> bool:
        """Return whether the client is connected."""
        return self._connected
        
    @property
    def people_detected(self) -> bool:
        """Alias for human_detected for compatibility."""
        return self.human_detected

    @property
    def pets_detected(self) -> bool:
        """Alias for pet_detected for compatibility."""
        return self.pet_detected

    @property
    def people_count(self) -> int:
        """Alias for human_count for compatibility."""
        return self.human_count
        
    @property
    def last_update_time(self) -> float:
        """Return timestamp of last update."""
        return self._last_update.timestamp() if self._last_update else None

    @property
    def model_name(self) -> str:
        """Return the model name used by the detector."""
        return self._config.get("model", "yolo11l") if hasattr(self, "_config") else "yolo11l"

    @property
    def connection_status(self) -> str:
        """Return connection status."""
        return "connected" if self._connected else "disconnected"

    @property
    def available(self) -> bool:
        """Return whether the device is available."""
        return self.is_connected
        
    async def async_initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the client with configuration and connect to the server."""
        try:
            # Store config
            self._config = config
            
            # Connect to the processing server
            connected = await self.async_connect()
            if not connected:
                _LOGGER.warning("Failed to connect to YOLO Processing Server during initialization")
                return False
            
            # Send create_detector message to the server
            try:
                _LOGGER.info("Creating detector on server with config: %s", config)
                create_message = {
                    "type": "create_detector",
                    "detector_id": self.detector_id,
                    "config": config
                }
                if not await self._send_message(create_message):
                    _LOGGER.error("Failed to send create_detector message, connection may be broken")
                    self._reconnect_event.set()
                    return False
                
                # Wait for response (using a state update request as a check)
                await asyncio.sleep(1)  # Give server time to process
                if not await self._send_message({"type": "get_state"}):
                    _LOGGER.error("Failed to send get_state message after detector creation")
                    self._reconnect_event.set()
                    return False
                
                _LOGGER.info("Detector created successfully on server")
            except Exception as create_ex:
                _LOGGER.error("Failed to create detector on server: %s", create_ex)
                # Continue anyway - the connection is established
            
            _LOGGER.info("YOLO Processing client initialized successfully")
            return True
            
        except Exception as ex:
            _LOGGER.exception("Failed to initialize YOLO client: %s", ex)
            return False
            
    async def async_shutdown(self) -> None:
        """Shut down the client and close all connections."""
        await self.async_close()