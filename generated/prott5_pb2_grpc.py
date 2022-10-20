# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import prott5_pb2 as prott5__pb2


class GenerativeSamplerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.SeqToEmbedding = channel.unary_unary(
                '/bionemo.model.protein.prott5nv.grpc.GenerativeSampler/SeqToEmbedding',
                request_serializer=prott5__pb2.InputSpec.SerializeToString,
                response_deserializer=prott5__pb2.OutputSpec.FromString,
                )


class GenerativeSamplerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def SeqToEmbedding(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_GenerativeSamplerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'SeqToEmbedding': grpc.unary_unary_rpc_method_handler(
                    servicer.SeqToEmbedding,
                    request_deserializer=prott5__pb2.InputSpec.FromString,
                    response_serializer=prott5__pb2.OutputSpec.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'bionemo.model.protein.prott5nv.grpc.GenerativeSampler', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class GenerativeSampler(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def SeqToEmbedding(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/bionemo.model.protein.prott5nv.grpc.GenerativeSampler/SeqToEmbedding',
            prott5__pb2.InputSpec.SerializeToString,
            prott5__pb2.OutputSpec.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
