// Reconnecting websocket client. Parses {type, data} envelopes and forwards
// them to the app; reports connection status for the top-bar indicator.

export class WSClient {
  constructor(url, { onMessage, onStatus }) {
    this.url = url;
    this.onMessage = onMessage;
    this.onStatus = onStatus;
    this.ws = null;
    this._retry = 0;
  }

  connect() {
    this.onStatus?.('connecting');
    const ws = new WebSocket(this.url);
    this.ws = ws;

    ws.onopen = () => { this._retry = 0; this.onStatus?.('online'); };
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      this.onMessage?.(msg);
    };
    ws.onclose = () => {
      this.onStatus?.('offline');
      const delay = Math.min(4000, 400 * 2 ** this._retry++);
      setTimeout(() => this.connect(), delay);
    };
    ws.onerror = () => ws.close();
  }
}
