"""Config flow for YOLO Presence Detection integration."""
import logging
import re
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
import homeassistant.helpers.config_validation as cv

from .const import (
    DOMAIN,
    CONF_STREAM_URL,
    CONF_DETECTION_INTERVAL,
    CONF_CONFIDENCE_THRESHOLD,
    CONF_INPUT_SIZE,
    CONF_MODEL,
    CONF_PROCESSING_SERVER,
    DEFAULT_NAME,
    DEFAULT_DETECTION_INTERVAL_CPU,
    DEFAULT_DETECTION_INTERVAL_GPU,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_INPUT_SIZE,
    DEFAULT_MODEL,
    DEFAULT_FRAME_SKIP_RATE_CPU,
    DEFAULT_FRAME_SKIP_RATE_GPU,
    DEFAULT_PROCESSING_SERVER,
    MODEL_OPTIONS,
    INPUT_SIZE_OPTIONS,
)

_LOGGER = logging.getLogger(__name__)


async def validate_stream_url(hass: HomeAssistant, stream_url: str) -> tuple[bool, str]:
    """Test if the stream URL can be accessed."""
    # In the new architecture, we don't need to validate the stream URL
    # as this will be handled by the processing server
    return True, "Stream will be validated by the processing server"


async def validate_processing_server(hass: HomeAssistant, server_url: str) -> tuple[bool, str]:
    """Test if the processing server is accessible."""
    from aiohttp import ClientSession, ClientError
    import asyncio
    
    try:
        async with ClientSession() as session:
            try:
                async with session.get(f"{server_url.rstrip('/')}/api/status", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "running":
                            return True, "Success"
                        else:
                            return False, "Server responded but is not in running state"
                    else:
                        return False, f"Server responded with status code {response.status}"
            except asyncio.TimeoutError:
                return False, "Connection timed out"
            except ClientError as ex:
                return False, f"Connection error: {str(ex)}"
    except Exception as ex:
        return False, f"Error connecting to server: {str(ex)}"


class YoloPresenceConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for YOLO Presence Detection."""

    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get the options flow for this handler."""
        return YoloPresenceOptionsFlow(config_entry)

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        errors = {}
        
        if user_input is not None:
            # Validate processing server URL
            valid_server, server_message = await validate_processing_server(
                self.hass, user_input[CONF_PROCESSING_SERVER]
            )
            if not valid_server:
                errors[CONF_PROCESSING_SERVER] = server_message
            
            if not errors:
                # Check if entry with this URL already exists
                existing_entries = [
                    entry.data[CONF_STREAM_URL]
                    for entry in self._async_current_entries()
                ]
                
                if user_input[CONF_STREAM_URL] in existing_entries:
                    errors[CONF_STREAM_URL] = "stream_already_configured"
                else:
                    return self.async_create_entry(
                        title=user_input[CONF_NAME], 
                        data=user_input
                    )

        # Set up default values - these are conservative since actual 
        # processing will happen in a separate container
        default_detection_interval = DEFAULT_DETECTION_INTERVAL_CPU
        default_frame_skip_rate = DEFAULT_FRAME_SKIP_RATE_CPU
        
        # Provide defaults in the data_schema
        data_schema = vol.Schema({
            vol.Required(CONF_NAME, default=DEFAULT_NAME): str,
            vol.Required(CONF_PROCESSING_SERVER, default=DEFAULT_PROCESSING_SERVER): str,
            vol.Required(CONF_STREAM_URL): str,
            vol.Required(CONF_MODEL, default=DEFAULT_MODEL): vol.In(MODEL_OPTIONS),
            vol.Required(CONF_DETECTION_INTERVAL, default=default_detection_interval): 
                vol.All(vol.Coerce(int), vol.Range(min=1, max=300)),
            vol.Required(CONF_CONFIDENCE_THRESHOLD, default=DEFAULT_CONFIDENCE_THRESHOLD): 
                vol.All(vol.Coerce(float), vol.Range(min=0.1, max=0.9)),
            vol.Required(CONF_INPUT_SIZE, default=DEFAULT_INPUT_SIZE): 
                vol.In(INPUT_SIZE_OPTIONS),
        })

        # Prepare message about architecture
        compatibility_message = (
            "This integration requires a separate YOLO processing server running PyTorch. "
            "See documentation for setup instructions."
        )
            
        return self.async_show_form(
            step_id="user", 
            data_schema=data_schema,
            errors=errors,
            description_placeholders={
                "architecture_info": compatibility_message
            }
        )


class YoloPresenceOptionsFlow(config_entries.OptionsFlow):
    """Handle options."""

    def __init__(self, config_entry):
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        errors = {}

        if user_input is not None:
            # Validate processing server URL if changed
            if (CONF_PROCESSING_SERVER in user_input and
                user_input[CONF_PROCESSING_SERVER] != self.config_entry.data.get(CONF_PROCESSING_SERVER)):
                valid_server, server_message = await validate_processing_server(
                    self.hass, user_input[CONF_PROCESSING_SERVER]
                )
                if not valid_server:
                    errors[CONF_PROCESSING_SERVER] = server_message

            if not errors:
                # Update the config entry
                return self.async_create_entry(title="", data=user_input)

        # Prepare current settings
        current_settings = {**self.config_entry.data, **self.config_entry.options}
        
        data_schema = vol.Schema({
            vol.Required(
                CONF_PROCESSING_SERVER, 
                default=current_settings.get(CONF_PROCESSING_SERVER, DEFAULT_PROCESSING_SERVER)
            ): str,
            vol.Required(
                CONF_STREAM_URL, 
                default=current_settings.get(CONF_STREAM_URL)
            ): str,
            vol.Required(
                CONF_MODEL, 
                default=current_settings.get(CONF_MODEL, DEFAULT_MODEL)
            ): vol.In(MODEL_OPTIONS),
            vol.Required(
                CONF_DETECTION_INTERVAL, 
                default=current_settings.get(CONF_DETECTION_INTERVAL, DEFAULT_DETECTION_INTERVAL_CPU)
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=300)),
            vol.Required(
                CONF_CONFIDENCE_THRESHOLD, 
                default=current_settings.get(CONF_CONFIDENCE_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD)
            ): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=0.9)),
            vol.Required(
                CONF_INPUT_SIZE, 
                default=current_settings.get(CONF_INPUT_SIZE, DEFAULT_INPUT_SIZE)
            ): vol.In(INPUT_SIZE_OPTIONS),
        })

        return self.async_show_form(
            step_id="init", 
            data_schema=data_schema,
            errors=errors,
        )