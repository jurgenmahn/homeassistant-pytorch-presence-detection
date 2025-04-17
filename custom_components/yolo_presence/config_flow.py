"""Config flow for YOLO Presence Detection integration."""

import logging
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant, callback

from .const import (
    DOMAIN,
    CONF_STREAM_URL,
    CONF_DETECTION_INTERVAL,
    CONF_CONFIDENCE_THRESHOLD,
    CONF_INPUT_SIZE,
    CONF_MODEL,
    CONF_PROCESSING_SERVER,
    CONF_PROCESSING_SERVER_PORT,
    CONF_USE_TCP_CONNECTION,
    CONF_USE_AUTO_OPTIMIZATION,
    CONF_FRAME_SKIP_RATE,
    CONF_DETECTION_FRAME_COUNT,
    CONF_CONSISTENT_DETECTION_COUNT,
    DEFAULT_NAME,
    DEFAULT_DETECTION_INTERVAL_CPU,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_INPUT_SIZE,
    DEFAULT_MODEL,
    DEFAULT_FRAME_SKIP_RATE_CPU,
    DEFAULT_DETECTION_FRAME_COUNT,
    DEFAULT_CONSISTENT_DETECTION_COUNT,
    DEFAULT_PROCESSING_SERVER,
    DEFAULT_PROCESSING_SERVER_PORT,
    DEFAULT_USE_TCP_CONNECTION,
    DEFAULT_USE_AUTO_OPTIMIZATION,
    MODEL_OPTIONS,
    INPUT_SIZE_OPTIONS,
)

_LOGGER = logging.getLogger(__name__)


async def validate_stream_url(hass: HomeAssistant, stream_url: str) -> tuple[bool, str]:
    """Test if the stream URL can be accessed."""
    # In the new architecture, we don't need to validate the stream URL
    # as this will be handled by the processing server
    return True, "Stream will be validated by the processing server"


async def validate_processing_server(
    hass: HomeAssistant, server_host: str
) -> tuple[bool, str]:
    """Test if the processing server is accessible via TCP socket."""
    import asyncio
    import socket

    try:
        # Get port from config or use default
        port = hass.data.get("port", 5505)

        # Try to establish a connection to check if the server is reachable
        future = asyncio.open_connection(server_host, port)

        try:
            reader, writer = await asyncio.wait_for(future, timeout=5)
            writer.close()
            await writer.wait_closed()
            return True, "Success"
        except asyncio.TimeoutError:
            return False, "Connection timed out"
        except (ConnectionRefusedError, socket.gaierror) as ex:
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
                        title=user_input[CONF_NAME], data=user_input
                    )

        # Set up default values - these are conservative since actual
        # processing will happen in a separate container
        default_detection_interval = DEFAULT_DETECTION_INTERVAL_CPU

        # Check if auto-optimization is enabled
        use_auto_optimization = DEFAULT_USE_AUTO_OPTIMIZATION
        if user_input and CONF_USE_AUTO_OPTIMIZATION in user_input:
            use_auto_optimization = user_input[CONF_USE_AUTO_OPTIMIZATION]

        # Build schema based on auto-optimization setting
        schema = {
            vol.Required(CONF_NAME, default=DEFAULT_NAME): str,
            vol.Required(
                CONF_PROCESSING_SERVER, default=DEFAULT_PROCESSING_SERVER
            ): str,
            vol.Required(
                CONF_PROCESSING_SERVER_PORT, default=DEFAULT_PROCESSING_SERVER_PORT
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=65535)),
            vol.Required(
                CONF_USE_TCP_CONNECTION, default=DEFAULT_USE_TCP_CONNECTION
            ): bool,
            vol.Required(
                CONF_STREAM_URL,
                default=user_input.get(CONF_STREAM_URL, "") if user_input else "",
            ): str,
            vol.Required(
                CONF_USE_AUTO_OPTIMIZATION, default=use_auto_optimization
            ): bool,
        }

        # Add manual configuration fields or disabled fields based on auto-optimization
        if not use_auto_optimization:
            schema.update(
                {
                    vol.Required(CONF_MODEL, default=DEFAULT_MODEL): vol.In(
                        MODEL_OPTIONS
                    ),
                    vol.Required(
                        CONF_DETECTION_INTERVAL, default=default_detection_interval
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=300)),
                    vol.Required(
                        CONF_CONFIDENCE_THRESHOLD, default=DEFAULT_CONFIDENCE_THRESHOLD
                    ): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=0.9)),
                    vol.Required(CONF_INPUT_SIZE, default=DEFAULT_INPUT_SIZE): vol.In(
                        INPUT_SIZE_OPTIONS
                    ),
                    vol.Required(
                        CONF_FRAME_SKIP_RATE, default=DEFAULT_FRAME_SKIP_RATE_CPU
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=30)),
                    vol.Required(
                        CONF_DETECTION_FRAME_COUNT,
                        default=DEFAULT_DETECTION_FRAME_COUNT,
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=25)),
                    vol.Required(
                        CONF_CONSISTENT_DETECTION_COUNT,
                        default=DEFAULT_CONSISTENT_DETECTION_COUNT,
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=25)),
                }
            )
        else:
            # Add fields with default values, but mark them as disabled
            schema.update(
                {
                    vol.Required(
                        CONF_MODEL,
                        default=DEFAULT_MODEL,
                        description={"disabled": True},
                    ): vol.In(MODEL_OPTIONS),
                    vol.Required(
                        CONF_DETECTION_INTERVAL,
                        default=DEFAULT_DETECTION_INTERVAL_CPU,
                        description={"disabled": True},
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=300)),
                    vol.Required(
                        CONF_CONFIDENCE_THRESHOLD,
                        default=DEFAULT_CONFIDENCE_THRESHOLD,
                        description={"disabled": True},
                    ): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=0.9)),
                    vol.Required(
                        CONF_INPUT_SIZE,
                        default=DEFAULT_INPUT_SIZE,
                        description={"disabled": True},
                    ): vol.In(INPUT_SIZE_OPTIONS),
                    vol.Required(
                        CONF_FRAME_SKIP_RATE,
                        default=DEFAULT_FRAME_SKIP_RATE_CPU,
                        description={"disabled": True},
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=30)),
                    vol.Required(
                        CONF_DETECTION_FRAME_COUNT,
                        default=DEFAULT_DETECTION_FRAME_COUNT,
                        description={"disabled": True},
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=25)),
                    vol.Required(
                        CONF_CONSISTENT_DETECTION_COUNT,
                        default=DEFAULT_CONSISTENT_DETECTION_COUNT,
                        description={"disabled": True},
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=25)),
                }
            )

        data_schema = vol.Schema(schema)

        # Prepare messages
        compatibility_message = (
            "This integration requires a separate YOLO processing server running PyTorch. "
            "See documentation for setup instructions."
        )

        optimization_info = (
            "When auto-optimization is enabled, the server will automatically adjust detection "
            "settings based on available resources. The detector will dynamically optimize "
            "model, resolution, frame rate, and other parameters for best performance on "
            "your hardware. Leave this disabled if you want full manual control."
        )

        return self.async_show_form(
            step_id="user",
            data_schema=data_schema,
            errors=errors,
            description_placeholders={
                "architecture_info": compatibility_message,
                "optimization_info": optimization_info,
            },
        )


class YoloPresenceOptionsFlow(config_entries.OptionsFlow):
    """Handle options."""

    def __init__(self, config_entry):
        """Initialize options flow."""
        # Don't save the config_entry reference directly
        self._entry_id = config_entry.entry_id
        self._entry_data = {**config_entry.data}
        self._entry_options = {**config_entry.options}

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        errors = {}

        if user_input is not None:
            # Validate processing server URL if changed
            if CONF_PROCESSING_SERVER in user_input and user_input[
                CONF_PROCESSING_SERVER
            ] != self._entry_data.get(CONF_PROCESSING_SERVER):
                valid_server, server_message = await validate_processing_server(
                    self.hass, user_input[CONF_PROCESSING_SERVER]
                )
                if not valid_server:
                    errors[CONF_PROCESSING_SERVER] = server_message

            if not errors:
                # Update the config entry
                return self.async_create_entry(title="", data=user_input)

        # Prepare current settings
        current_settings = {**self._entry_data, **self._entry_options}

        # Check if auto-optimization is enabled
        use_auto_optimization = current_settings.get(
            CONF_USE_AUTO_OPTIMIZATION, DEFAULT_USE_AUTO_OPTIMIZATION
        )

        # Get the schema based on auto-optimization setting
        if user_input is not None and CONF_USE_AUTO_OPTIMIZATION in user_input:
            # User is changing the auto-optimization setting in this request
            use_auto_optimization = user_input[CONF_USE_AUTO_OPTIMIZATION]

        # Build the schema dynamically based on whether auto-optimization is enabled
        schema = {
            vol.Required(
                CONF_PROCESSING_SERVER,
                default=current_settings.get(
                    CONF_PROCESSING_SERVER, DEFAULT_PROCESSING_SERVER
                ),
            ): str,
            vol.Required(
                CONF_PROCESSING_SERVER_PORT,
                default=current_settings.get(
                    CONF_PROCESSING_SERVER_PORT, DEFAULT_PROCESSING_SERVER_PORT
                ),
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=65535)),
            vol.Required(
                CONF_USE_TCP_CONNECTION,
                default=current_settings.get(
                    CONF_USE_TCP_CONNECTION, DEFAULT_USE_TCP_CONNECTION
                ),
            ): bool,
            vol.Required(
                CONF_STREAM_URL, default=current_settings.get(CONF_STREAM_URL)
            ): str,
            vol.Required(
                CONF_USE_AUTO_OPTIMIZATION,
                default=use_auto_optimization,
                description={"suggested_value": use_auto_optimization},
            ): bool,
        }

        # Add the manual configuration fields if auto-optimization is disabled
        if not use_auto_optimization:
            schema.update(
                {
                    vol.Required(
                        CONF_MODEL,
                        default=current_settings.get(CONF_MODEL, DEFAULT_MODEL),
                    ): vol.In(MODEL_OPTIONS),
                    vol.Required(
                        CONF_DETECTION_INTERVAL,
                        default=current_settings.get(
                            CONF_DETECTION_INTERVAL, DEFAULT_DETECTION_INTERVAL_CPU
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=300)),
                    vol.Required(
                        CONF_CONFIDENCE_THRESHOLD,
                        default=current_settings.get(
                            CONF_CONFIDENCE_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD
                        ),
                    ): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=0.9)),
                    vol.Required(
                        CONF_INPUT_SIZE,
                        default=current_settings.get(
                            CONF_INPUT_SIZE, DEFAULT_INPUT_SIZE
                        ),
                    ): vol.In(INPUT_SIZE_OPTIONS),
                    vol.Required(
                        CONF_FRAME_SKIP_RATE,
                        default=current_settings.get(
                            CONF_FRAME_SKIP_RATE, DEFAULT_FRAME_SKIP_RATE_CPU
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=30)),
                    vol.Required(
                        CONF_DETECTION_FRAME_COUNT,
                        default=current_settings.get(
                            CONF_DETECTION_FRAME_COUNT, DEFAULT_DETECTION_FRAME_COUNT
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=25)),
                    vol.Required(
                        CONF_CONSISTENT_DETECTION_COUNT,
                        default=current_settings.get(
                            CONF_CONSISTENT_DETECTION_COUNT,
                            DEFAULT_CONSISTENT_DETECTION_COUNT,
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=25)),
                }
            )
        else:
            # Add fields with default values, but mark them as disabled
            # This approach lets us show the defaults but disable user input
            schema.update(
                {
                    vol.Required(
                        CONF_MODEL,
                        default=DEFAULT_MODEL,
                        description={"disabled": True},
                    ): vol.In(MODEL_OPTIONS),
                    vol.Required(
                        CONF_DETECTION_INTERVAL,
                        default=DEFAULT_DETECTION_INTERVAL_CPU,
                        description={"disabled": True},
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=300)),
                    vol.Required(
                        CONF_CONFIDENCE_THRESHOLD,
                        default=DEFAULT_CONFIDENCE_THRESHOLD,
                        description={"disabled": True},
                    ): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=0.9)),
                    vol.Required(
                        CONF_INPUT_SIZE,
                        default=DEFAULT_INPUT_SIZE,
                        description={"disabled": True},
                    ): vol.In(INPUT_SIZE_OPTIONS),
                    vol.Required(
                        CONF_FRAME_SKIP_RATE,
                        default=DEFAULT_FRAME_SKIP_RATE_CPU,
                        description={"disabled": True},
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=30)),
                    vol.Required(
                        CONF_DETECTION_FRAME_COUNT,
                        default=DEFAULT_DETECTION_FRAME_COUNT,
                        description={"disabled": True},
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=25)),
                    vol.Required(
                        CONF_CONSISTENT_DETECTION_COUNT,
                        default=DEFAULT_CONSISTENT_DETECTION_COUNT,
                        description={"disabled": True},
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=25)),
                }
            )

        data_schema = vol.Schema(schema)

        # Define explanation text about auto-optimization
        optimization_info = (
            "When auto-optimization is enabled, the server will automatically adjust detection "
            "settings based on available resources. The detector will dynamically optimize "
            "model, resolution, frame rate, and other parameters for best performance on "
            "your hardware. Leave this disabled if you want full manual control."
        )

        return self.async_show_form(
            step_id="init",
            data_schema=data_schema,
            errors=errors,
            description_placeholders={"optimization_info": optimization_info},
        )
