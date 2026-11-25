"""
List DirectShow video device names on Windows, in the same order that
cv2.VideoCapture(index, cv2.CAP_DSHOW) numbers them.

Uses only ctypes (no extra packages), so it also works inside the
PyInstaller build. On other platforms, or if anything fails, it returns [].
"""
import ctypes
import sys
from ctypes import wintypes

_CLSID_SYSTEM_DEVICE_ENUM = "{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}"
_IID_ICREATE_DEV_ENUM = "{29840822-5B84-11D0-BD3B-00A0C911CE86}"
_CLSID_VIDEO_INPUT_CATEGORY = "{860BB310-5D01-11d0-BD3B-00A0C911CE86}"
_IID_IPROPERTY_BAG = "{55272A00-42CB-11CE-8135-00AA004BB851}"

_CLSCTX_INPROC_SERVER = 0x1
_COINIT_APARTMENTTHREADED = 0x2
_VT_BSTR = 8

# vtable slots
_RELEASE = 2
_CREATE_CLASS_ENUMERATOR = 3  # ICreateDevEnum
_ENUM_NEXT = 3                # IEnumMoniker
_BIND_TO_STORAGE = 9          # IMoniker
_PROPBAG_READ = 3             # IPropertyBag


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _VARIANT(ctypes.Structure):
    _fields_ = [
        ("vt", ctypes.c_ushort),
        ("r1", ctypes.c_ushort),
        ("r2", ctypes.c_ushort),
        ("r3", ctypes.c_ushort),
        ("data", ctypes.c_void_p * 2),
    ]


def _guid(text):
    g = _GUID()
    hr = ctypes.windll.ole32.CLSIDFromString(ctypes.c_wchar_p(text), ctypes.byref(g))
    if hr != 0:
        raise OSError(f"CLSIDFromString failed: {hr:#x}")
    return g


def _method(obj, slot, *argtypes):
    """Return a callable for vtable slot `slot` of COM object pointer `obj`."""
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.c_void_p))[0]
    fn_addr = ctypes.cast(vtbl, ctypes.POINTER(ctypes.c_void_p))[slot]
    proto = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, *argtypes)
    fn = proto(fn_addr)
    return lambda *args: fn(obj, *args)


def _release(obj):
    if obj:
        vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.c_void_p))[0]
        fn_addr = ctypes.cast(vtbl, ctypes.POINTER(ctypes.c_void_p))[_RELEASE]
        ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(fn_addr)(obj)


def _read_friendly_name(moniker, iid_bag):
    bag = ctypes.c_void_p()
    try:
        _method(moniker, _BIND_TO_STORAGE,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p))(
            None, None, ctypes.byref(iid_bag), ctypes.byref(bag))
    except OSError:
        return None

    var = _VARIANT()
    try:
        _method(bag, _PROPBAG_READ, ctypes.c_wchar_p, ctypes.POINTER(_VARIANT), ctypes.c_void_p)(
            "FriendlyName", ctypes.byref(var), None)
        if var.vt == _VT_BSTR and var.data[0]:
            return ctypes.wstring_at(var.data[0])
        return None
    except OSError:
        return None
    finally:
        ctypes.windll.oleaut32.VariantClear(ctypes.byref(var))
        _release(bag)


def list_video_devices():
    """Return DirectShow video device names; index i matches CAP_DSHOW index i."""
    if sys.platform != "win32":
        return []

    ole32 = ctypes.windll.ole32
    hr_init = ole32.CoInitializeEx(None, _COINIT_APARTMENTTHREADED)
    # S_OK (0) / S_FALSE (1) must be balanced by CoUninitialize;
    # RPC_E_CHANGED_MODE means COM is already set up on this thread, which is fine.
    needs_uninit = hr_init in (0, 1)

    dev_enum = ctypes.c_void_p()
    enum_moniker = ctypes.c_void_p()
    names = []
    try:
        clsid_sde = _guid(_CLSID_SYSTEM_DEVICE_ENUM)
        iid_cde = _guid(_IID_ICREATE_DEV_ENUM)
        cat = _guid(_CLSID_VIDEO_INPUT_CATEGORY)
        iid_bag = _guid(_IID_IPROPERTY_BAG)

        hr = ole32.CoCreateInstance(ctypes.byref(clsid_sde), None, _CLSCTX_INPROC_SERVER,
                                    ctypes.byref(iid_cde), ctypes.byref(dev_enum))
        if hr != 0:
            return []

        hr = _method(dev_enum, _CREATE_CLASS_ENUMERATOR,
                     ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p), wintypes.DWORD)(
            ctypes.byref(cat), ctypes.byref(enum_moniker), 0)
        if hr != 0 or not enum_moniker:
            return []  # S_FALSE: no video devices

        enum_next = _method(enum_moniker, _ENUM_NEXT,
                            wintypes.ULONG, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.ULONG))
        while True:
            moniker = ctypes.c_void_p()
            fetched = wintypes.ULONG(0)
            if enum_next(1, ctypes.byref(moniker), ctypes.byref(fetched)) != 0 or fetched.value == 0:
                break
            try:
                names.append(_read_friendly_name(moniker, iid_bag) or "Unknown camera")
            finally:
                _release(moniker)
        return names
    except Exception:
        return names
    finally:
        _release(enum_moniker)
        _release(dev_enum)
        if needs_uninit:
            ole32.CoUninitialize()


_MF_VERSION = 0x00020070
_MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE = "{c60ac5fe-252a-478f-a0ef-bc8fa5f7cad3}"
_MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID = "{8ac3587a-4ae7-42d8-99e0-0a6013eef90f}"
_MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME = "{60d0e559-52f8-4fa2-bbce-acdb34a8ec01}"
_ATTR_GET_ALLOCATED_STRING = 13  # IMFAttributes / IMFActivate
_ATTR_SET_GUID = 24


def list_msmf_video_devices():
    """Return Media Foundation video device names; index i matches CAP_MSMF index i."""
    if sys.platform != "win32":
        return []

    ole32 = ctypes.windll.ole32
    hr_init = ole32.CoInitializeEx(None, _COINIT_APARTMENTTHREADED)
    needs_uninit = hr_init in (0, 1)
    names = []
    attrs = ctypes.c_void_p()
    devices = ctypes.POINTER(ctypes.c_void_p)()
    count = ctypes.c_uint32(0)
    mfplat = mf = None
    started = False
    try:
        mfplat = ctypes.windll.mfplat
        mf = ctypes.windll.mf
        if mfplat.MFStartup(_MF_VERSION, 0) != 0:
            return []
        started = True

        if mfplat.MFCreateAttributes(ctypes.byref(attrs), 1) != 0:
            return []
        key = _guid(_MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE)
        value = _guid(_MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID)
        _method(attrs, _ATTR_SET_GUID, ctypes.POINTER(_GUID), ctypes.POINTER(_GUID))(
            ctypes.byref(key), ctypes.byref(value))

        if mf.MFEnumDeviceSources(attrs, ctypes.byref(devices), ctypes.byref(count)) != 0:
            return []

        name_key = _guid(_MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME)
        for i in range(count.value):
            activate = devices[i]
            text = ctypes.c_wchar_p()
            length = ctypes.c_uint32(0)
            try:
                _method(activate, _ATTR_GET_ALLOCATED_STRING,
                        ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_wchar_p), ctypes.POINTER(ctypes.c_uint32))(
                    ctypes.byref(name_key), ctypes.byref(text), ctypes.byref(length))
                names.append(text.value or "Unknown camera")
            except OSError:
                names.append("Unknown camera")
            finally:
                if text:
                    ole32.CoTaskMemFree(text)
                _release(activate)
        return names
    except Exception:
        return names
    finally:
        if devices:
            ole32.CoTaskMemFree(devices)
        _release(attrs)
        if started:
            mfplat.MFShutdown()
        if needs_uninit:
            ole32.CoUninitialize()


def is_windows_virtual_camera(name):
    """True for Windows 'connected cameras' (e.g. a phone linked via Link to Windows)."""
    lowered = name.lower()
    return "windows virtual camera" in lowered or "windows 가상 카메라" in lowered


if __name__ == "__main__":
    print("DirectShow:")
    for i, name in enumerate(list_video_devices()):
        print(" ", i, name)
    print("Media Foundation:")
    for i, name in enumerate(list_msmf_video_devices()):
        print(" ", i, name)
