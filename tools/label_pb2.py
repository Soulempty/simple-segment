# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: label.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0blabel.proto\x12\x07\x61q.aidi\x1a\x19google/protobuf/any.proto\"\x1f\n\x07Point2f\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\"\'\n\x06Size2f\x12\r\n\x05width\x18\x01 \x01(\x02\x12\x0e\n\x06height\x18\x02 \x01(\x02\"\xe2\x01\n\x08KeyPoint\x12\"\n\x08location\x18\x01 \x01(\x0b\x32\x10.aq.aidi.Point2f\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\r\n\x05score\x18\x03 \x01(\x02\x12\r\n\x05\x61ngle\x18\x04 \x01(\x02\x12\x0e\n\x06radius\x18\x05 \x01(\x02\x12\x30\n\x08\x65xt_info\x18\x0f \x03(\x0b\x32\x1e.aq.aidi.KeyPoint.ExtInfoEntry\x1a\x44\n\x0c\x45xtInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12#\n\x05value\x18\x02 \x01(\x0b\x32\x14.google.protobuf.Any:\x02\x38\x01\"(\n\x04Ring\x12 \n\x06points\x18\x01 \x03(\x0b\x32\x10.aq.aidi.Point2f\"F\n\x07Polygon\x12\x1c\n\x05outer\x18\x01 \x01(\x0b\x32\r.aq.aidi.Ring\x12\x1d\n\x06inners\x18\x02 \x03(\x0b\x32\r.aq.aidi.Ring\"\xe5\x01\n\x06Region\x12!\n\x07polygon\x18\x01 \x01(\x0b\x32\x10.aq.aidi.Polygon\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\r\n\x05score\x18\x03 \x01(\x02\x12%\n\nkey_points\x18\x04 \x03(\x0b\x32\x11.aq.aidi.KeyPoint\x12.\n\x08\x65xt_info\x18\x0f \x03(\x0b\x32\x1c.aq.aidi.Region.ExtInfoEntry\x1a\x44\n\x0c\x45xtInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12#\n\x05value\x18\x02 \x01(\x0b\x32\x14.google.protobuf.Any:\x02\x38\x01\"\xd7\x03\n\x05Label\x12\x30\n\x0c\x64\x61taset_type\x18\x01 \x01(\x0e\x32\x1a.aq.aidi.Label.DataSetType\x12!\n\x08img_size\x18\x02 \x01(\x0b\x32\x0f.aq.aidi.Size2f\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\r\n\x05score\x18\x04 \x01(\x02\x12 \n\x07regions\x18\x05 \x03(\x0b\x32\x0f.aq.aidi.Region\x12\x1f\n\x05masks\x18\x06 \x03(\x0b\x32\x10.aq.aidi.Polygon\x12#\n\thardcases\x18\x07 \x03(\x0b\x32\x10.aq.aidi.Polygon\x12+\n\rorigin_result\x18\x08 \x01(\x0b\x32\x14.google.protobuf.Any\x12-\n\x08\x65xt_info\x18\x0f \x03(\x0b\x32\x1b.aq.aidi.Label.ExtInfoEntry\x1a\x44\n\x0c\x45xtInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12#\n\x05value\x18\x02 \x01(\x0b\x32\x14.google.protobuf.Any:\x02\x38\x01\"R\n\x0b\x44\x61taSetType\x12\x0b\n\x07Unknown\x10\x00\x12\x0b\n\x07Segment\x10\x01\x12\r\n\tDetection\x10\x02\x12\x0c\n\x08\x43lassify\x10\x03\x12\x0c\n\x08Location\x10\x04\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'label_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _KEYPOINT_EXTINFOENTRY._options = None
  _KEYPOINT_EXTINFOENTRY._serialized_options = b'8\001'
  _REGION_EXTINFOENTRY._options = None
  _REGION_EXTINFOENTRY._serialized_options = b'8\001'
  _LABEL_EXTINFOENTRY._options = None
  _LABEL_EXTINFOENTRY._serialized_options = b'8\001'
  _POINT2F._serialized_start=51
  _POINT2F._serialized_end=82
  _SIZE2F._serialized_start=84
  _SIZE2F._serialized_end=123
  _KEYPOINT._serialized_start=126
  _KEYPOINT._serialized_end=352
  _KEYPOINT_EXTINFOENTRY._serialized_start=284
  _KEYPOINT_EXTINFOENTRY._serialized_end=352
  _RING._serialized_start=354
  _RING._serialized_end=394
  _POLYGON._serialized_start=396
  _POLYGON._serialized_end=466
  _REGION._serialized_start=469
  _REGION._serialized_end=698
  _REGION_EXTINFOENTRY._serialized_start=284
  _REGION_EXTINFOENTRY._serialized_end=352
  _LABEL._serialized_start=701
  _LABEL._serialized_end=1172
  _LABEL_EXTINFOENTRY._serialized_start=284
  _LABEL_EXTINFOENTRY._serialized_end=352
  _LABEL_DATASETTYPE._serialized_start=1090
  _LABEL_DATASETTYPE._serialized_end=1172
# @@protoc_insertion_point(module_scope)
