"""
Hikvision Camera Control via ISAPI
Handles camera configuration for thermal, infrared, and RGB cameras
"""

import requests
from requests.auth import HTTPDigestAuth
import json
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class HikvisionCamera:
    """Control Hikvision cameras via ISAPI interface"""

    def __init__(self, ip: str, username: str, password: str, camera_type: str = "rgb"):
        self.ip = ip
        self.base = f"http://{ip}/ISAPI"
        self.auth = HTTPDigestAuth(username, password)
        self.camera_type = camera_type

        self.endpoints = {
            "Image": "/Image/channels/1",
            "IrcutFilter": "/Image/channels/1/IrcutFilter",
            "IRLight": "/System/Video/inputs/channels/1/overlays/infraredLight",
            "Exposure": "/Image/channels/1/Exposure",
            "WhiteBalance": "/Image/channels/1/WhiteBalance",
            "Color": "/Image/channels/1/Color",
            "Sharpness": "/Image/channels/1/Sharpness",
            "NoiseReduce": "/Image/channels/1/NoiseReduce",
            "WDR": "/Image/channels/1/WDR",
            "BLC": "/Image/channels/1/BLC"
        }

    def get_config(self, endpoint_name: str) -> Optional[str]:
        """Get current configuration for an endpoint"""
        endpoint = self.endpoints.get(endpoint_name)
        if not endpoint:
            logger.error(f"Unknown endpoint: {endpoint_name}")
            return None

        try:
            r = requests.get(self.base + endpoint, auth=self.auth, timeout=5)
            if r.status_code == 200:
                return r.text
            else:
                logger.error(f"Error getting {endpoint_name}: {r.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection error getting {endpoint_name}: {e}")
            return None

    def set_config(self, endpoint_name: str, xml_data: str) -> bool:
        """Set configuration for an endpoint"""
        endpoint = self.endpoints.get(endpoint_name)
        if not endpoint:
            logger.error(f"Unknown endpoint: {endpoint_name}")
            return False

        headers = {'Content-Type': 'application/xml'}
        try:
            r = requests.put(
                self.base + endpoint,
                data=xml_data,
                auth=self.auth,
                headers=headers,
                timeout=5
            )

            if r.status_code == 200:
                logger.info(f"✓ Successfully updated {endpoint_name} on {self.ip}")
                return True
            else:
                logger.error(f"✗ Error updating {endpoint_name}: {r.status_code}")
                logger.error(r.text)
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection error setting {endpoint_name}: {e}")
            return False

    def set_ircut_filter(self, filter_type: str = "auto") -> bool:
        """
        Set IR cut filter
        Args:
            filter_type: auto, day, or night
        """
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<IrcutFilter version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
    <IrcutFilterType>{filter_type}</IrcutFilterType>
    <nightToDayFilterLevel>4</nightToDayFilterLevel>
    <nightToDayFilterTime>5</nightToDayFilterTime>
</IrcutFilter>"""
        return self.set_config("IrcutFilter", xml)

    def set_exposure(self, exposure_type: str = "auto") -> bool:
        """
        Set exposure settings
        Args:
            exposure_type: auto or manual
        """
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Exposure version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
    <ExposureType>{exposure_type}</ExposureType>
    <OverexposeSuppress>
        <enabled>false</enabled>
        <hightLightDistanceLevel>50</hightLightDistanceLevel>
        <lowLightDistanceLevel>50</lowLightDistanceLevel>
    </OverexposeSuppress>
</Exposure>"""
        return self.set_config("Exposure", xml)

    def set_white_balance(self, style: str = "auto1", red: int = 50, blue: int = 50) -> bool:
        """
        Set white balance
        Args:
            style: auto1, auto2, manual, outdoor, indoor, fluorescent
            red: 0-100
            blue: 0-100
        """
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<WhiteBalance version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
    <WhiteBalanceStyle>{style}</WhiteBalanceStyle>
    <WhiteBalanceRed>{red}</WhiteBalanceRed>
    <WhiteBalanceBlue>{blue}</WhiteBalanceBlue>
</WhiteBalance>"""
        return self.set_config("WhiteBalance", xml)

    def set_color(self, brightness: int = 50, contrast: int = 50, saturation: int = 50) -> bool:
        """
        Set color parameters
        Args:
            brightness: 0-100
            contrast: 0-100
            saturation: 0-100
        """
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Color version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
    <brightnessLevel>{brightness}</brightnessLevel>
    <contrastLevel>{contrast}</contrastLevel>
    <saturationLevel>{saturation}</saturationLevel>
</Color>"""
        return self.set_config("Color", xml)

    def set_sharpness(self, level: int = 50) -> bool:
        """
        Set sharpness level
        Args:
            level: 0-100
        """
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Sharpness version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
    <SharpnessLevel>{level}</SharpnessLevel>
</Sharpness>"""
        return self.set_config("Sharpness", xml)

    def set_noise_reduce(self, mode: str = "general", level: int = 50) -> bool:
        """
        Set noise reduction
        Args:
            mode: close, general, expert
            level: 0-100
        """
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<NoiseReduce version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
    <mode>{mode}</mode>
    <GeneralMode>
        <generalLevel>{level}</generalLevel>
    </GeneralMode>
</NoiseReduce>"""
        return self.set_config("NoiseReduce", xml)

    def set_wdr(self, mode: str = "close", level: int = 50) -> bool:
        """
        Set WDR (Wide Dynamic Range)
        Args:
            mode: close or open
            level: 0-100
        """
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<WDR version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
    <mode>{mode}</mode>
    <WDRLevel>{level}</WDRLevel>
</WDR>"""
        return self.set_config("WDR", xml)

    def set_blc(self, enabled: bool = False) -> bool:
        """
        Set BLC (Back Light Compensation)
        Args:
            enabled: True or False
        """
        enabled_str = "true" if enabled else "false"
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<BLC version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
    <enabled>{enabled_str}</enabled>
</BLC>"""
        return self.set_config("BLC", xml)

    def apply_profile(self, profile_settings: Dict[str, Any]) -> bool:
        """
        Apply a complete profile from camera_profiles.json
        Args:
            profile_settings: The hikvision_settings dict from a profile
        """
        logger.info(f"Applying profile to {self.camera_type} camera at {self.ip}")

        success = True

        # IR Cut Filter
        if "ircut_filter" in profile_settings:
            success &= self.set_ircut_filter(profile_settings["ircut_filter"])

        # Exposure
        if "exposure_type" in profile_settings:
            success &= self.set_exposure(profile_settings["exposure_type"])

        # White Balance
        if "white_balance" in profile_settings:
            success &= self.set_white_balance(profile_settings["white_balance"])

        # Color
        if "color" in profile_settings:
            color = profile_settings["color"]
            success &= self.set_color(
                color.get("brightness", 50),
                color.get("contrast", 50),
                color.get("saturation", 50)
            )

        # Sharpness
        if "sharpness" in profile_settings:
            success &= self.set_sharpness(profile_settings["sharpness"])

        # Noise Reduction
        if "noise_reduce" in profile_settings:
            nr = profile_settings["noise_reduce"]
            success &= self.set_noise_reduce(nr.get("mode", "general"), nr.get("level", 50))

        # WDR
        if "wdr" in profile_settings:
            wdr = profile_settings["wdr"]
            success &= self.set_wdr(wdr.get("mode", "close"), wdr.get("level", 50))

        # BLC
        if "blc" in profile_settings:
            success &= self.set_blc(profile_settings["blc"].get("enabled", False))

        return success

    def get_all_settings(self) -> Dict[str, str]:
        """Get all current camera settings"""
        settings = {}
        for name in self.endpoints.keys():
            config = self.get_config(name)
            if config:
                settings[name] = config
        return settings


def configure_camera_from_profile(camera_config: Dict, profile_config: Dict) -> bool:
    """
    Configure a camera using settings from config files
    Args:
        camera_config: Camera entry from cameras.json
        profile_config: Profile entry from camera_profiles.json
    """
    connection = camera_config["connection"]

    camera = HikvisionCamera(
        ip=connection["ip"],
        username=connection["username"],
        password=connection["password"],
        camera_type=camera_config["type"]
    )

    if "hikvision_settings" in profile_config:
        return camera.apply_profile(profile_config["hikvision_settings"])
    else:
        logger.warning(f"No hikvision_settings found in profile for {camera_config['name']}")
        return False
