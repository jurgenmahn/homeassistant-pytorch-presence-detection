"""API client for communicating with the YOLO Presence Processing Server."""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, Callable, List
import aiohttp
from homeassistant.core import HomeAssistant
import homeassistant.util.dt as dt_util
from .const import ATTR_DEVICE_ID

_LOGGER = logging.getLogger(__name__)


class YoloProcessingApiClient:
    """API client for the YOLO Presence Processing Server.

    Uses simple HTTP requests to poll the processing server.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        server_url: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 5505,
        detector_id: Optional[str] = None,
        update_interval: int = 30,
        server_port: Optional[int] = None,
        use_tcp: bool = True,  # kept for compatibility
    ) -> None:
        """Initialize the API client."""
        self.hass = hass

        # Handle backward compatibility and different initialization methods
        # For HTTP, we need a full URL
        if server_url:
            # If server_url doesn't include http://, add it
            if not server_url.startswith("http://") and not server_url.startswith(
                "https://"
            ):
                self.base_url = f"http://{server_url}"
            else:
                self.base_url = server_url
        else:
            self.base_url = f"http://{host}"

        # Port priority: server_port param > port param > default
        port_to_use = server_port or port

        # Update URL with port if specified
        if ":" not in self.base_url.split("/")[2]:
            self.base_url = f"{self.base_url}:{port_to_use}"

        self.detector_id = detector_id
        self.update_interval = update_interval

        _LOGGER.info("Initializing YOLO HTTP client for %s with update interval: %s seconds", 
                    self.base_url, self.update_interval)

        # State tracking
        self._human_detected = False
        self._pet_detected = False
        self._human_count = 0
        self._pet_count = 0
        self._last_update = None
        self._available = False
        self._last_error = None
        self._session = None

        # Callback management
        self._update_callbacks: List[Callable[[], None]] = []

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a client session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        assert self._session is not None
        return self._session

    async def async_update(self) -> None:
        """Update the state by polling the server."""
        try:
            session = await self._get_session()

            # Prepare the poll data with detector configuration
            poll_data = {
                "detector_id": self.detector_id,
                "config": {
                    "name": f"HA Detector {self.detector_id}",
                    "detection_interval": self.update_interval,
                    # Add any other configuration needed by the detector
                },
            }

            # Make sure stream_url is in config if we have it
            config = getattr(self, "_config", {})
            if config and "stream_url" in config:
                poll_data["config"]["stream_url"] = config["stream_url"]

                # Add other config options if they exist
                for key in [
                    "model",
                    "input_size",
                    "confidence_threshold",
                    "frame_skip_rate",
                    "use_auto_optimization",  # Add auto_optimization flag
                ]:
                    if key in config:
                        poll_data["config"][key] = config[key]
                
                # Ensure update_interval from configuration is correctly passed
                if "detection_interval" in config:
                    poll_data["config"]["detection_interval"] = config["detection_interval"]

            # Send the poll request
            _LOGGER.debug("Polling YOLO server at %s with data: %s", self.base_url, poll_data)
            poll_url = f"{self.base_url}/poll"

            async with session.post(poll_url, json=poll_data) as response:
                response_status = response.status
                response_text = await response.text()
                
                # Log the complete request/response cycle
                _LOGGER.debug("Poll response (HTTP %s): %s", response_status, response_text)
                
                if response_status == 200:
                    try:
                        result = json.loads(response_text)
                        
                        if result.get("status") == "success":
                            # Process the state
                            state = result.get("state", {})
                            _LOGGER.debug("Received state update: %s", state)
                            await self._handle_state_update(state)
                            self._available = True
                            self._last_error = None
                        else:
                            _LOGGER.error(
                                "Error from server: %s",
                                result.get("message", "Unknown error"),
                            )
                            self._available = False
                            self._last_error = result.get("message", "Unknown error")
                    except json.JSONDecodeError as err:
                        _LOGGER.error(
                            "Failed to parse server response: %s. Raw response: %s", 
                            err, response_text[:200]
                        )
                        self._available = False
                        self._last_error = f"JSON parse error: {err}"
                else:
                    _LOGGER.error(
                        "Failed to poll server: HTTP %s %s. Response: %s", 
                        response_status, response.reason, response_text[:200]
                    )
                    self._available = False
                    self._last_error = f"HTTP {response_status} {response.reason}"

        except aiohttp.ClientError as err:
            _LOGGER.error("Connection error: %s", err)
            self._available = False
            self._last_error = str(err)
        except asyncio.TimeoutError:
            _LOGGER.error("Connection timeout")
            self._available = False
            self._last_error = "Connection timeout"
        except Exception as ex:
            _LOGGER.exception("Unexpected error updating from server: %s", ex)
            self._available = False
            self._last_error = str(ex)

    async def _handle_state_update(self, state: Dict[str, Any]) -> None:
        """Handle a state update."""
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

    async def async_initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the client with configuration."""
        try:
            self._config = config

            # Store initialization parameters in our configuration
            self._config["update_interval"] = self.update_interval

            # Get initial state from server
            await self.async_update()

            _LOGGER.info("YOLO Processing client initialized successfully")
            return True
        except Exception as ex:
            _LOGGER.exception("Failed to initialize YOLO client: %s", ex)
            return False

    async def async_shutdown(self) -> None:
        """Shut down the client and clean up resources."""
        try:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None

            # Optionally tell the server to shut down this detector
            try:
                session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))

                shutdown_data = {"detector_id": self.detector_id}

                async with session.post(
                    f"{self.base_url}/shutdown", json=shutdown_data
                ) as response:
                    if response.status == 200:
                        _LOGGER.info("Successfully shut down detector on server")
                    else:
                        _LOGGER.warning(
                            "Failed to shut down detector on server: HTTP %s",
                            response.status,
                        )
            except Exception as ex:
                _LOGGER.warning("Error shutting down detector on server: %s", ex)
            finally:
                if session and not session.closed:
                    await session.close()

            _LOGGER.info("API client closed")
        except Exception as ex:
            _LOGGER.error("Error shutting down client: %s", ex)

    def register_update_callback(
        self, callback: Callable[[], None]
    ) -> Callable[[], None]:
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
        return self._available

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
    def last_update_time(self) -> Optional[float]:
        """Return timestamp of last update."""
        return self._last_update.timestamp() if self._last_update else None

    @property
    def model_name(self) -> str:
        """Return the model name used by the detector."""
        return (
            self._config.get("model", "yolo11l")
            if hasattr(self, "_config")
            else "yolo11l"
        )

    @property
    def connection_status(self) -> str:
        """Return connection status."""
        return "connected" if self._available else "disconnected"

    @property
    def available(self) -> bool:
        """Return whether the device is available."""
        return self._available
