"""API client for communicating with the YOLO Presence Processing Server."""
import asyncio
import logging
import time
from typing import Dict, Any, Optional

import aiohttp
from aiohttp.client_exceptions import ClientConnectorError, ClientError
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
        
        # Polling task
        self._polling_task = None
        self._stop_event = asyncio.Event()
        
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
                
                # Start polling for status updates
                await self._start_polling()
                return True
            else:
                _LOGGER.error(f"Failed to initialize detector: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as ex:
            _LOGGER.error(f"Error initializing connection to processing server: {str(ex)}")
            return False

    async def _start_polling(self) -> None:
        """Start polling the server for updates."""
        if self._polling_task is not None:
            return
            
        self._stop_event.clear()
        self._polling_task = asyncio.create_task(self._polling_loop())
        _LOGGER.debug(f"Started polling task for detector {self.detector_id}")

    async def _polling_loop(self) -> None:
        """Poll the server for updates in a loop."""
        last_successful_update = 0
        
        while not self._stop_event.is_set():
            try:
                # Get detector status from server
                detector_status = await self._api_call("GET", f"/api/detectors/{self.detector_id}")
                
                if detector_status and "error" not in detector_status:
                    # Update state from server response
                    old_connection_status = self.connection_status
                    old_people_detected = self.people_detected
                    old_pets_detected = self.pets_detected
                    old_people_count = self.people_count
                    old_pet_count = self.pet_count
                    
                    # Update local state
                    self.is_running = detector_status.get("is_running", False)
                    self.connection_status = detector_status.get("connection_status", "unknown")
                    self.people_detected = detector_status.get("people_detected", False)
                    self.pets_detected = detector_status.get("pets_detected", False)
                    self.people_count = detector_status.get("people_count", 0)
                    self.pet_count = detector_status.get("pet_count", 0)
                    self.model_name = detector_status.get("model", "unknown")
                    self.device = detector_status.get("device", "cpu")
                    
                    # Only update timestamp after first successful update
                    server_update_time = detector_status.get("last_update_time", 0)
                    if server_update_time > self.last_update_time:
                        self.last_update_time = server_update_time
                        
                        # If anything important changed, call the update callbacks
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
                    
                    last_successful_update = time.time()
                else:
                    # Handle error response
                    error_message = detector_status.get("error", "Unknown error")
                    _LOGGER.warning(f"Error getting detector status: {error_message}")
                    
                    # If we haven't had a successful update for 30 seconds, mark as disconnected
                    if time.time() - last_successful_update > 30:
                        if self.connection_status != "server_disconnected":
                            self.connection_status = "server_disconnected"
                            await self._fire_connection_event("server_disconnected")
                            self._call_update_callbacks()
            
            except ClientConnectorError:
                # Connection to server failed
                if self.connection_status != "server_unavailable":
                    self.connection_status = "server_unavailable"
                    await self._fire_connection_event("server_unavailable")
                    self._call_update_callbacks()
                    
            except Exception as ex:
                _LOGGER.error(f"Error in polling loop: {str(ex)}")
                
            # Wait for next update
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.update_interval)
            except asyncio.TimeoutError:
                # This is expected - just continue with the next update
                pass

    async def async_shutdown(self) -> None:
        """Shutdown the client and delete the detector on the server."""
        # Stop polling
        if self._polling_task is not None:
            self._stop_event.set()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None
        
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