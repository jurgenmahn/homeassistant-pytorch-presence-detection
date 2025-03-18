"""API client for communicating with the YOLO Presence Processing Server."""
import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Callable

import aiohttp
from aiohttp.client_exceptions import ClientConnectorError, ClientError, ClientPayloadError
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
import homeassistant.util.dt as dt_util

from .const import ATTR_DEVICE_ID

_LOGGER = logging.getLogger(__name__)

class YoloProcessingApiClient:
    """API client for the YOLO Presence Processing Server."""

    def __init__(
        self, 
        hass: HomeAssistant, 
        server_url: str, 
        detector_id: str,
        update_interval: int = 1
    ):
        """Initialize the API client.
        
        Args:
            hass: Home Assistant instance
            server_url: URL of the processing server (e.g., "http://192.168.1.100:5000")
            detector_id: ID of the detector to monitor
            update_interval: How often to poll the server for updates (seconds)
        """
        self.hass = hass
        self.server_url = server_url.rstrip("/")  # Remove trailing slash if present
        self.detector_id = detector_id
        self.update_interval = update_interval
        
        # State variables
        self.is_initialized = False
        self.is_running = False
        self.connection_status = "disconnected"
        self.last_update_time = 0
        self.people_detected = False
        self.pets_detected = False
        self.people_count = 0
        self.pet_count = 0
        self.model_name = None
        self.device = None
        
        # SSE connection
        self._sse_task = None
        self._stop_event = asyncio.Event()
        self._reconnect_delay = 1  # Initial reconnect delay in seconds
        self._max_reconnect_delay = 60  # Maximum reconnect delay
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10  # Max number of reconnect attempts before giving up
        
        # Callback handling
        self.update_callbacks = []

    async def async_initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the connection to the processing server.
        
        Args:
            config: Configuration for creating the detector on the server
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Prepare configuration for server
            server_config = dict(config)
            server_config["detector_id"] = self.detector_id
            
            # Create detector on server
            result = await self._api_call("POST", "/api/detectors", json=server_config)
            
            if result.get("success"):
                self.is_initialized = True
                _LOGGER.info(f"Successfully initialized detector {self.detector_id} on server {self.server_url}")
                
                # Start SSE connection for real-time updates
                await self._start_sse_connection()
                return True
            else:
                error_msg = result.get("error", "Unknown error")
                
                # Check if the error is because the detector already exists
                if "already exists" in error_msg:
                    _LOGGER.warning(f"Detector {self.detector_id} already exists on server, attempting to reconnect")
                    
                    # Verify detector exists by getting its status
                    status = await self._api_call("GET", f"/api/detectors/{self.detector_id}")
                    if "error" not in status:
                        # Detector exists and is accessible, so we can use it
                        self.is_initialized = True
                        _LOGGER.info(f"Successfully reconnected to existing detector {self.detector_id}")
                        
                        # Start SSE connection for real-time updates
                        await self._start_sse_connection()
                        return True
                    else:
                        _LOGGER.error(f"Detector exists but cannot be accessed: {status.get('error')}")
                        return False
                else:
                    _LOGGER.error(f"Failed to initialize detector: {error_msg}")
                    return False
                
        except Exception as ex:
            _LOGGER.error(f"Error initializing connection to processing server: {str(ex)}")
            return False

    async def _start_sse_connection(self) -> None:
        """Start SSE connection for real-time updates."""
        if self._sse_task is not None:
            return
            
        self._stop_event.clear()
        self._reconnect_attempts = 0
        self._reconnect_delay = 1
        self._sse_task = asyncio.create_task(self._sse_connection_loop())
        _LOGGER.debug(f"Started SSE connection for detector {self.detector_id}")

    async def _sse_connection_loop(self) -> None:
        """Maintain SSE connection with automatic reconnect."""
        while not self._stop_event.is_set() and self._reconnect_attempts < self._max_reconnect_attempts:
            try:
                await self._connect_to_sse()
                
                # If we get here, the connection closed normally
                # Reset connection retries on successful connection
                self._reconnect_attempts = 0
                self._reconnect_delay = 1
                
                if not self._stop_event.is_set():
                    _LOGGER.info(f"SSE connection closed, reconnecting for {self.detector_id}...")
                    # Small delay before reconnect
                    await asyncio.sleep(1)
                
            except (ClientConnectorError, ClientPayloadError, asyncio.TimeoutError):
                # Connection failed, attempt to reconnect with backoff
                if not self._stop_event.is_set():
                    self._reconnect_attempts += 1
                    
                    if self.connection_status != "server_unavailable":
                        self.connection_status = "server_unavailable"
                        await self._fire_connection_event("server_unavailable")
                        self._call_update_callbacks()
                    
                    if self._reconnect_attempts < self._max_reconnect_attempts:
                        _LOGGER.warning(
                            f"SSE connection failed (attempt {self._reconnect_attempts}/{self._max_reconnect_attempts}), "
                            f"reconnecting in {self._reconnect_delay}s..."
                        )
                        
                        # Wait for reconnect delay (or until stop event)
                        try:
                            await asyncio.wait_for(self._stop_event.wait(), timeout=self._reconnect_delay)
                        except asyncio.TimeoutError:
                            pass
                        
                        # Increase reconnect delay with exponential backoff (max 60s)
                        self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
                    else:
                        _LOGGER.error(
                            f"SSE connection failed after {self._max_reconnect_attempts} attempts. "
                            f"Giving up for detector {self.detector_id}."
                        )
                        self.connection_status = "server_disconnected"
                        await self._fire_connection_event("server_disconnected")
                        self._call_update_callbacks()
            
            except Exception as ex:
                _LOGGER.error(f"Unexpected error in SSE connection: {str(ex)}")
                if not self._stop_event.is_set():
                    # Wait a bit before retry
                    await asyncio.sleep(5)
        
        _LOGGER.debug(f"SSE connection loop ended for detector {self.detector_id}")

    async def _connect_to_sse(self) -> None:
        """Connect to the SSE endpoint and process events."""
        session = async_get_clientsession(self.hass)
        
        try:
            async with session.get(
                f"{self.server_url}/api/detectors/{self.detector_id}/events",
                timeout=30,
                headers={"Accept": "text/event-stream"}
            ) as response:
                if response.status != 200:
                    response_text = await response.text()
                    _LOGGER.error(f"SSE connection failed with status {response.status}: {response_text}")
                    raise ClientError(f"SSE connection failed with status {response.status}")
                
                # Connection established successfully
                if self.connection_status == "server_unavailable":
                    self.connection_status = "connected"
                    await self._fire_connection_event("connected")
                    self._call_update_callbacks()
                
                _LOGGER.debug(f"SSE connection established for {self.detector_id}")
                
                # Process the event stream
                buffer = ""
                
                # Implementation based on aiohttp SSE client
                async for line_bytes in response.content:
                    # Reset reconnection attempts on successful data reception
                    if self._reconnect_attempts > 0:
                        self._reconnect_attempts = 0
                        self._reconnect_delay = 1
                        
                    line = line_bytes.decode('utf8')
                    buffer += line
                    
                    # Process complete messages (ending with double newline)
                    if buffer.endswith('\n\n'):
                        messages = buffer.split('\n\n')
                        buffer = ""
                        
                        for message in messages:
                            if not message:
                                continue
                                
                            # Process each line in the message
                            data = None
                            event_type = None
                            
                            for msg_line in message.split('\n'):
                                msg_line = msg_line.strip()
                                
                                # Skip empty lines and comments/keepalives
                                if not msg_line or msg_line.startswith(':'):
                                    continue
                                
                                if msg_line.startswith('data:'):
                                    try:
                                        data = json.loads(msg_line[5:].strip())
                                    except json.JSONDecodeError as ex:
                                        _LOGGER.warning(f"Invalid JSON in SSE message: {ex}")
                                elif msg_line.startswith('event:'):
                                    event_type = msg_line[6:].strip()
                            
                            # Process the message
                            if data:
                                await self._handle_sse_message(data)
                            elif event_type and event_type in ('shutdown', 'detector_shutdown'):
                                _LOGGER.info(f"Received {event_type} event from server")
                                # Connection will be closed by server, we'll reconnect automatically
                    
                    # Check if we need to stop
                    if self._stop_event.is_set():
                        break
                        
                _LOGGER.debug(f"SSE stream ended for {self.detector_id}")
        except Exception as ex:
            _LOGGER.error(f"Error in SSE connection for {self.detector_id}: {str(ex)}")
            raise

    async def _handle_sse_message(self, data: Dict[str, Any]) -> None:
        """Process an SSE message containing detector state."""
        if not data:
            return
            
        # Update state from server response
        old_connection_status = self.connection_status
        old_people_detected = self.people_detected
        old_pets_detected = self.pets_detected
        old_people_count = self.people_count
        old_pet_count = self.pet_count
        
        # Update local state
        self.is_running = data.get("is_running", self.is_running)
        self.connection_status = data.get("connection_status", self.connection_status)
        self.people_detected = data.get("people_detected", self.people_detected)
        self.pets_detected = data.get("pets_detected", self.pets_detected)
        self.people_count = data.get("people_count", self.people_count)
        self.pet_count = data.get("pet_count", self.pet_count)
        self.model_name = data.get("model", self.model_name)
        self.device = data.get("device", self.device)
        
        # Update timestamp
        server_update_time = data.get("last_update_time", 0)
        if server_update_time > self.last_update_time:
            self.last_update_time = server_update_time
        
        # If anything important changed, call the update callbacks and fire events
        if (old_connection_status != self.connection_status or
            old_people_detected != self.people_detected or
            old_pets_detected != self.pets_detected or
            old_people_count != self.people_count or
            old_pet_count != self.pet_count):
            
            self._call_update_callbacks()
            
            # Fire events for significant changes
            if old_connection_status != self.connection_status:
                await self._fire_connection_event(self.connection_status)
                
            if old_people_detected != self.people_detected:
                await self._fire_human_event(self.people_detected)
                
            if old_pets_detected != self.pets_detected:
                await self._fire_pet_event(self.pets_detected)
                
            if old_people_count != self.people_count:
                await self._fire_human_count_event(self.people_count)
                
            if old_pet_count != self.pet_count:
                await self._fire_pet_count_event(self.pet_count)

    async def async_shutdown(self) -> None:
        """Shutdown the client and delete the detector on the server."""
        # Stop SSE connection
        if self._sse_task is not None:
            self._stop_event.set()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass
            self._sse_task = None
        
        # Delete detector on server if we're initialized
        if self.is_initialized:
            try:
                await self._api_call("DELETE", f"/api/detectors/{self.detector_id}")
                _LOGGER.info(f"Successfully deleted detector {self.detector_id} from server")
            except Exception as ex:
                _LOGGER.warning(f"Error deleting detector from server: {str(ex)}")

    async def _api_call(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an API call to the server.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint (e.g., "/api/detectors")
            **kwargs: Additional arguments to pass to the aiohttp request
            
        Returns:
            Dict containing the response JSON or an error description
        """
        url = f"{self.server_url}{endpoint}"
        
        session = async_get_clientsession(self.hass)
        
        try:
            async with session.request(method, url, **kwargs) as response:
                if response.status in (200, 201):
                    # Check if response is JSON or binary (image)
                    content_type = response.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        return await response.json()
                    elif "image/" in content_type:
                        # For image responses
                        return {"image": True, "content": await response.read()}
                    else:
                        return {"success": True}
                else:
                    try:
                        error_json = await response.json()
                        return {"error": error_json.get("error", f"HTTP {response.status}")}
                    except:
                        return {"error": f"HTTP {response.status}"}
        except asyncio.TimeoutError:
            return {"error": "Request timed out"}
        except ClientConnectorError:
            return {"error": "Could not connect to server"}
        except ClientError as ex:
            return {"error": f"Client error: {str(ex)}"}
        except Exception as ex:
            return {"error": f"Unexpected error: {str(ex)}"}

    def register_update_callback(self, callback_func):
        """Register a callback for state updates."""
        self.update_callbacks.append(callback_func)
        return lambda: self.update_callbacks.remove(callback_func)

    def _call_update_callbacks(self):
        """Call all registered callbacks."""
        for callback_func in self.update_callbacks:
            callback_func()

    async def _fire_human_event(self, is_detected: bool) -> None:
        """Fire a Home Assistant event when human detection state changes."""
        from .const import DOMAIN, EVENT_HUMAN_DETECTED
        
        data = {
            ATTR_DEVICE_ID: self.detector_id,
            "name": self.detector_id,
            "detected": is_detected,
        }
        self.hass.bus.async_fire(EVENT_HUMAN_DETECTED, data)
        _LOGGER.debug(f"{self.detector_id}: Human detected state changed to {is_detected}")

    async def _fire_pet_event(self, is_detected: bool) -> None:
        """Fire a Home Assistant event when pet detection state changes."""
        from .const import DOMAIN, EVENT_PET_DETECTED
        
        data = {
            ATTR_DEVICE_ID: self.detector_id,
            "name": self.detector_id,
            "detected": is_detected,
        }
        self.hass.bus.async_fire(EVENT_PET_DETECTED, data)
        _LOGGER.debug(f"{self.detector_id}: Pet detected state changed to {is_detected}")

    async def _fire_human_count_event(self, count: int) -> None:
        """Fire a Home Assistant event when human count changes."""
        from .const import DOMAIN, EVENT_HUMAN_COUNT_CHANGED
        
        data = {
            ATTR_DEVICE_ID: self.detector_id,
            "name": self.detector_id,
            "count": count,
        }
        self.hass.bus.async_fire(EVENT_HUMAN_COUNT_CHANGED, data)
        _LOGGER.debug(f"{self.detector_id}: Human count changed to {count}")

    async def _fire_pet_count_event(self, count: int) -> None:
        """Fire a Home Assistant event when pet count changes."""
        from .const import DOMAIN, EVENT_PET_COUNT_CHANGED
        
        data = {
            ATTR_DEVICE_ID: self.detector_id,
            "name": self.detector_id,
            "count": count,
        }
        self.hass.bus.async_fire(EVENT_PET_COUNT_CHANGED, data)
        _LOGGER.debug(f"{self.detector_id}: Pet count changed to {count}")

    async def _fire_connection_event(self, status: str) -> None:
        """Fire a Home Assistant event when connection status changes."""
        from .const import DOMAIN, EVENT_CONNECTION_STATUS_CHANGED
        
        data = {
            ATTR_DEVICE_ID: self.detector_id,
            "name": self.detector_id,
            "status": status,
        }
        self.hass.bus.async_fire(EVENT_CONNECTION_STATUS_CHANGED, data)
        _LOGGER.info(f"{self.detector_id}: Connection status changed to {status}")

    @property
    def available(self) -> bool:
        """Return if the API client is available."""
        return self.connection_status == "connected"