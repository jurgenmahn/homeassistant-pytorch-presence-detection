"""Config flow for YOLO Presence Detection integration."""
import logging
import re
import voluptuous as vol
import cv2

from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
import homeassistant.helpers.config_validation as cv

# Try to import torch, but handle if not available
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

from .const import (
    DOMAIN,
    CONF_STREAM_URL,
    CONF_DETECTION_INTERVAL,
    CONF_CONFIDENCE_THRESHOLD,
    CONF_INPUT_SIZE,
    CONF_MODEL,
    CONF_FRAME_SKIP_RATE,
    DEFAULT_NAME,
    DEFAULT_DETECTION_INTERVAL_CPU,
    DEFAULT_DETECTION_INTERVAL_GPU,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_INPUT_SIZE,
    DEFAULT_MODEL,
    DEFAULT_FRAME_SKIP_RATE_CPU,
    DEFAULT_FRAME_SKIP_RATE_GPU,
    MODEL_OPTIONS,
    INPUT_SIZE_OPTIONS,
)

_LOGGER = logging.getLogger(__name__)


async def validate_stream_url(hass: HomeAssistant, stream_url: str) -> tuple[bool, str]:
    """Test if the stream URL can be accessed."""
    def _test_stream():
        try:
            # Just try to open the stream without actually reading frames
            # This is much more lightweight for validation
            cap = cv2.VideoCapture(stream_url)
            is_opened = cap.isOpened()
            is_readable = is_opened  # Assume readable if it opens
            cap.release()
            return is_opened, is_readable
        except Exception as ex:
            return False, f"Error: {str(ex)}"

    try:
        is_opened, is_readable = await hass.async_add_executor_job(_test_stream)
        if not is_opened:
            return False, "Could not connect to stream URL"
        if not is_readable:
            return False, "Could connect but not read from stream"
        return True, "Success"
    except Exception as ex:
        return False, f"Error accessing stream: {str(ex)}"


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

        # Detect if CUDA is available for better defaults
        has_cuda = False
        if TORCH_AVAILABLE:
            try:
                has_cuda = torch.cuda.is_available()
            except Exception:
                has_cuda = False
        
        if user_input is not None:
            # Validate stream URL
            valid_url, url_message = await validate_stream_url(self.hass, user_input[CONF_STREAM_URL])
            if not valid_url:
                errors[CONF_STREAM_URL] = url_message
            
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

        # Set up default values
        default_detection_interval = (
            DEFAULT_DETECTION_INTERVAL_GPU if has_cuda else DEFAULT_DETECTION_INTERVAL_CPU
        )
        default_frame_skip_rate = (
            DEFAULT_FRAME_SKIP_RATE_GPU if has_cuda else DEFAULT_FRAME_SKIP_RATE_CPU
        )
        
        # Provide defaults in the data_schema
        data_schema = vol.Schema({
            vol.Required(CONF_NAME, default=DEFAULT_NAME): str,
            vol.Required(CONF_STREAM_URL): str,
            vol.Required(CONF_MODEL, default=DEFAULT_MODEL): vol.In(MODEL_OPTIONS),
            vol.Required(CONF_DETECTION_INTERVAL, default=default_detection_interval): 
                vol.All(vol.Coerce(int), vol.Range(min=1, max=300)),
            vol.Required(CONF_CONFIDENCE_THRESHOLD, default=DEFAULT_CONFIDENCE_THRESHOLD): 
                vol.All(vol.Coerce(float), vol.Range(min=0.1, max=0.9)),
            vol.Required(CONF_INPUT_SIZE, default=DEFAULT_INPUT_SIZE): 
                vol.In(INPUT_SIZE_OPTIONS),
            vol.Required(CONF_FRAME_SKIP_RATE, default=default_frame_skip_rate): 
                vol.All(vol.Coerce(int), vol.Range(min=1, max=20)),
        })

        # Prepare message about compatibility
        compatibility_message = ""
        if not TORCH_AVAILABLE:
            compatibility_message = (
                "PyTorch not installed. The integration will run in compatibility mode " 
                "with limited functionality. Install PyTorch manually for full features."
            )
        elif has_cuda:
            compatibility_message = "GPU detected! Using optimized defaults."
        else:
            compatibility_message = "No GPU detected. Using CPU optimized settings."
            
        return self.async_show_form(
            step_id="user", 
            data_schema=data_schema,
            errors=errors,
            description_placeholders={
                "has_gpu": compatibility_message
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
            if CONF_STREAM_URL in user_input:
                # Validate stream URL if changed
                if user_input[CONF_STREAM_URL] != self.config_entry.data.get(CONF_STREAM_URL):
                    valid_url, url_message = await validate_stream_url(
                        self.hass, user_input[CONF_STREAM_URL]
                    )
                    if not valid_url:
                        errors[CONF_STREAM_URL] = url_message

            if not errors:
                # Update the config entry
                return self.async_create_entry(title="", data=user_input)

        # Detect if CUDA is available for slider defaults
        has_cuda = False
        if TORCH_AVAILABLE:
            try:
                has_cuda = torch.cuda.is_available()
            except Exception:
                has_cuda = False
        
        # Prepare current settings
        current_settings = {**self.config_entry.data, **self.config_entry.options}
        
        data_schema = vol.Schema({
            vol.Required(CONF_STREAM_URL, default=current_settings.get(CONF_STREAM_URL)): str,
            vol.Required(CONF_MODEL, default=current_settings.get(CONF_MODEL, DEFAULT_MODEL)): 
                vol.In(MODEL_OPTIONS),
            vol.Required(
                CONF_DETECTION_INTERVAL, 
                default=current_settings.get(CONF_DETECTION_INTERVAL, 
                    DEFAULT_DETECTION_INTERVAL_GPU if has_cuda else DEFAULT_DETECTION_INTERVAL_CPU)
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=300)),
            vol.Required(
                CONF_CONFIDENCE_THRESHOLD, 
                default=current_settings.get(CONF_CONFIDENCE_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD)
            ): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=0.9)),
            vol.Required(
                CONF_INPUT_SIZE, 
                default=current_settings.get(CONF_INPUT_SIZE, DEFAULT_INPUT_SIZE)
            ): vol.In(INPUT_SIZE_OPTIONS),
            vol.Required(
                CONF_FRAME_SKIP_RATE, 
                default=current_settings.get(CONF_FRAME_SKIP_RATE, 
                    DEFAULT_FRAME_SKIP_RATE_GPU if has_cuda else DEFAULT_FRAME_SKIP_RATE_CPU)
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=20)),
        })

        return self.async_show_form(
            step_id="init", 
            data_schema=data_schema,
            errors=errors,
        )