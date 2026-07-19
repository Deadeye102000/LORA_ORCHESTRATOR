from fastapi import WebSocket
from typing import Dict, List

class ConnectionManager:
    def __init__(self):
        # Maps room_id (str) -> List[WebSocket]
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room_id: str):
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
        self.active_connections[room_id].append(websocket)
        print(f"[WebSocket] Client connected to room: {room_id} (Active connections: {len(self.active_connections[room_id])})")

    async def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.active_connections:
            if websocket in self.active_connections[room_id]:
                self.active_connections[room_id].remove(websocket)
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]
        print(f"[WebSocket] Client disconnected from room: {room_id}")

    async def broadcast_to_room(self, room_id: str, message: dict):
        if room_id in self.active_connections:
            # Use list copy to avoid mutation errors during iteration
            for connection in list(self.active_connections[room_id]):
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print(f"[WebSocket] Broadcast error in room {room_id}: {e}")
                    # Clean up dead connection
                    try:
                        self.active_connections[room_id].remove(connection)
                    except ValueError:
                        pass
