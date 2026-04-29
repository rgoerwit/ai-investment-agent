from __future__ import annotations


class MCPCallError(Exception):
    """Error during an MCP tool call."""

    def __init__(
        self,
        message: str,
        category: str,
        server_id: str,
        tool_name: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.server_id = server_id
        self.tool_name = tool_name
        self.retryable = retryable
