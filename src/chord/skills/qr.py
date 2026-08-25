"""QR codes, both directions.

Making one is local: ``qrcode`` draws it, the PNG rides out on the
attachment channel that already carries charts. Reading one needs a
decoder, and the choice there is the whole story of this module.

``pyzbar`` wants a system libzbar, which turns "pip install chord" into
"pip install chord, then apt install something". ``opencv`` can do it
inside a 60 MB wheel. ``zxing-cpp`` ships prebuilt wheels for every
platform this runs on, weighs a couple of megabytes, needs nothing from
the host, and reads the other barcode formats for free - so a photo of
a book's barcode answers too, without anyone having to ask for it.

The decoder never sees a URL the guard in :mod:`chord.skills._fetch` has
not cleared: a QR image is a link a chat user handed over, exactly like
the ones read_url opens.
"""

from __future__ import annotations

import io
import logging
from typing import ClassVar
from urllib.parse import urlsplit

import qrcode
import zxingcpp
from PIL import Image

from chord.attachments import attach
from chord.skills._fetch import fetch_image
from chord.skills._http import SkillHTTPError
from chord.skills.base import Skill

logger = logging.getLogger(__name__)

#: A QR code tops out near 3 kB of binary payload, and long before that
#: the modules get too dense for a phone camera to resolve off a screen.
MAX_QR_TEXT = 1000

#: Pixels per QR module. 10 gives a code that scans off a laptop screen
#: at arm's length without making the PNG enormous.
BOX_SIZE = 10

#: Quiet zone, in modules. Four is what the spec asks for, and scanners
#: really do fail without it.
BORDER = 4

#: Some decoders need more pixels than a screenshot has. One retry at
#: double size costs milliseconds and rescues small or blurry captures.
UPSCALE = 2

#: Pixels an image may hold before it is refused. A QR code photographed
#: on a phone is under 15 megapixels; a 50000x50000 PNG that compresses
#: to a few hundred kilobytes is a decompression bomb, and the size cap
#: on the download cannot see it because the bomb is small on the wire.
#: Checked from the header, before any pixels are allocated.
MAX_IMAGE_PIXELS = 40_000_000


def make_qr_png(text: str) -> bytes:
    """Render ``text`` as a QR code PNG.

    Black on white, deliberately. An inverted or tinted code looks
    better in a dark channel and scans worse everywhere, and a QR nobody
    can scan is decoration.
    """
    code = qrcode.QRCode(box_size=BOX_SIZE, border=BORDER)
    code.add_data(text)
    code.make(fit=True)
    image = code.make_image(fill_color="black", back_color="white")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def decode_barcodes(data: bytes) -> list[tuple[str, str]]:
    """``(format, text)`` for every barcode in an image."""
    try:
        image = Image.open(io.BytesIO(data))
        # open() reads the header only, so the dimensions are known
        # before anything is decoded - which is the whole point.
        width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise SkillHTTPError(
                f"That image is {width}x{height}, far larger than any QR code "
                "needs; refusing to decode it."
            )
        image.load()
    except SkillHTTPError:
        raise
    except Exception as exc:  # noqa: BLE001 - any unreadable image lands here
        raise SkillHTTPError(f"That does not open as an image ({exc}).") from exc

    rgb = image.convert("RGB")
    found = _read(rgb)
    if not found:
        # A screenshot of a code on a phone is often too small to
        # resolve; a plain upscale is enough surprisingly often.
        found = _read(rgb.resize((rgb.width * UPSCALE, rgb.height * UPSCALE)))
    return found


def _read(image: Image.Image) -> list[tuple[str, str]]:
    return [
        (result.format.name, result.text) for result in zxingcpp.read_barcodes(image) if result.text
    ]


def looks_like_url(text: str) -> bool:
    """Whether a decoded payload is something someone might click."""
    parts = urlsplit(text.strip())
    return parts.scheme in ("http", "https") and bool(parts.netloc)


class QrEncodeSkill(Skill):
    name = "make_qr"
    description = (
        "Make a QR code image from text or a link and post it to the "
        "channel (QR 만들어줘, QR코드로 뽑아줘). Takes any text: a URL, "
        "wifi details, a phone number, a message."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "What the QR code should contain, e.g. a URL.",
            }
        },
        "required": ["text"],
    }

    async def run(self, text: str) -> str:
        payload = (text or "").strip()
        if not payload:
            raise SkillHTTPError("There is nothing to put in the QR code.")
        if len(payload) > MAX_QR_TEXT:
            raise SkillHTTPError(
                f"That is {len(payload)} characters; a scannable QR code tops "
                f"out around {MAX_QR_TEXT}. Shorten it, or shorten the URL first."
            )

        try:
            png = make_qr_png(payload)
        except Exception as exc:  # noqa: BLE001 - reported to the model
            logger.exception("Could not render a QR code")
            raise SkillHTTPError(f"Could not draw that as a QR code ({exc}).") from exc

        if not attach("qr.png", png):
            raise SkillHTTPError("The QR image could not be attached to this reply.")
        return (
            f"QR code created for: {payload}\n"
            "The image is attached to this reply - point the user at it."
        )


class QrDecodeSkill(Skill):
    name = "read_qr"
    description = (
        "Read a QR code or barcode out of an image and return what it "
        "says (QR 뭐라고 써있어, 이거 읽어줘). Give it the image's URL - "
        "images posted in the channel arrive in the message as "
        "'[attached image: ...]'."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "image_url": {
                "type": "string",
                "description": "Direct link to the image holding the code.",
            }
        },
        "required": ["image_url"],
    }

    async def run(self, image_url: str) -> str:
        fetched = await fetch_image(image_url)
        found = decode_barcodes(fetched.body)
        if not found:
            raise SkillHTTPError(
                "No QR code or barcode was found in that image. If it is a "
                "photo, a sharper or closer crop usually works."
            )

        lines = []
        for symbol_format, text in found:
            lines.append(f"{symbol_format}: {text}")
            if looks_like_url(text):
                lines.append(
                    "  (this is a link - it was not opened; "
                    "check_url_safety can vet it before anyone clicks)"
                )
        return "\n".join(lines)
