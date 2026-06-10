#!/usr/bin/env python3
"""Zhihu social network graph visualization.

Builds a directed ego-network graph centered on a Zhihu user, showing:
- Followees (people the user follows)
- Followers (people who follow the user)
- Mutual follows (both directions)

Supports arbitrary-depth recursive expansion so you can see not just
your direct connections but also *their* connections, and so on.

Uses NetworkX for graph construction and Plotly for rendering.

Dependencies (pip install):
    requests  lxml  networkx  numpy  plotly  kaleido

Usage:
    python zhihu_social_graph.py \\
        --cookie "d_c0=...; z_c0=..." \\
        --user-agent "Mozilla/5.0 ..." \\
        --url-token your-url-token

    # Or read cookie from browser's cURL export format:
    python zhihu_social_graph.py --cookie "$(pbpaste)" --url-token someone
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import struct
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import requests
from lxml import html as lxml_html

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path.cwd() / "ZhihuSocialGraph"
MEMBER_API = "https://www.zhihu.com/api/v4/members/{token}"

# ═══════════════════════════════════════════════════════════════════════════════
# Wait / throttle utility
# ═══════════════════════════════════════════════════════════════════════════════


def _wait(seconds: float = 1.0) -> None:
    """Sleep with jitter to avoid hammering the API."""
    delay = max(0.1, random.lognormvariate(math.log(seconds), 0.5))
    time.sleep(min(delay, 3 * seconds))


# ═══════════════════════════════════════════════════════════════════════════════
# Zhihu API helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _stream_handler(
    session: requests.Session,
    initial_url: str,
    parser,
    delay: float = 1.0,
):
    """Paginate a Zhihu API endpoint, yielding parsed items one by one."""
    current_url = initial_url
    while current_url:
        resp = session.get(current_url)
        resp.raise_for_status()
        data = resp.json()
        paging = data.get("paging", {})
        for item in parser(data):
            yield item
        if paging.get("is_end", True):
            current_url = None
        else:
            current_url = (paging.get("next") or "").replace("http://", "https://")
        if current_url:
            _wait(delay)


def _parse_followee(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "user",
        "id": item.get("id", ""),
        "name": item.get("name", ""),
        "url_token": item.get("url_token", ""),
        "headline": item.get("headline", ""),
        "avatar_url": (item.get("avatar_url_template") or "").replace("{size}", "xl"),
        "gender": item.get("gender", -1),
        "follower_count": item.get("follower_count", 0),
        "answer_count": item.get("answer_count", 0),
        "articles_count": item.get("articles_count", 0),
        "is_following": item.get("is_following", False),
        "is_followed": item.get("is_followed", False),
        "url": f"https://www.zhihu.com/people/{item.get('url_token', '')}",
    }


def _parse_followees_list(data: dict[str, Any]):
    for item in data.get("data", []):
        yield _parse_followee(item)


def fetch_followees(
    session: requests.Session,
    url_token: str,
    limit: int = 20,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    url = (
        f"{MEMBER_API.format(token=url_token)}/followees"
        "?include=data%5B*%5D.answer_count%2Carticles_count%2Cgender"
        "%2Cfollower_count%2Cis_followed%2Cis_following"
        "%2Cbadge%5B%3F(type%3Dbest_answerer)%5D.topics"
        f"&offset=0&limit={limit}"
    )
    items: list[dict[str, Any]] = []
    for item in _stream_handler(session, url, _parse_followees_list):
        items.append(item)
        if max_items is not None and len(items) >= max_items:
            break
    return items


def fetch_followers(
    session: requests.Session,
    url_token: str,
    limit: int = 20,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    url = (
        f"{MEMBER_API.format(token=url_token)}/followers"
        "?include=data%5B*%5D.answer_count%2Carticles_count%2Cgender"
        "%2Cfollower_count%2Cis_followed%2Cis_following"
        "%2Cbadge%5B%3F(type%3Dbest_answerer)%5D.topics"
        f"&offset=0&limit={limit}"
    )
    items: list[dict[str, Any]] = []
    for item in _stream_handler(session, url, _parse_followees_list):
        items.append(item)
        if max_items is not None and len(items) >= max_items:
            break
    return items


def fetch_page_html(session: requests.Session, url: str) -> str:
    return session.get(url.replace("http://", "https://")).text


def get_page_state(html_text: str, key: str = "entities") -> dict[str, Any]:
    doc = lxml_html.fromstring(html_text)
    script_tag = doc.find(".//script[@id='js-initialData']")
    if script_tag is None or not script_tag.text:
        raise ValueError("Could not find 'js-initialData' script tag")
    initial_data = json.loads(script_tag.text)
    return initial_data["initialState"][key]


def fetch_member_profile(session: requests.Session, url_token: str) -> dict[str, Any] | None:
    """Fetch a member's public profile info (via HTML js-initialData)."""
    try:
        entities = get_page_state(
            fetch_page_html(session, f"https://www.zhihu.com/people/{url_token}")
        )
    except Exception:
        return None

    users = entities.get("users", {})
    user_data = users.get(url_token)
    if user_data is None:
        for v in users.values():
            if isinstance(v, dict) and v.get("urlToken") == url_token:
                user_data = v
                break
    if user_data is None:
        return None

    return {
        "id": user_data.get("id", ""),
        "name": user_data.get("name", ""),
        "url_token": user_data.get("urlToken", url_token),
        "headline": user_data.get("headline", ""),
        "avatar_url": (user_data.get("avatarUrlTemplate") or "").replace("{size}", "xl"),
        "gender": user_data.get("gender", -1),
        "follower_count": user_data.get("followerCount", 0),
        "following_count": user_data.get("followingCount", 0),
        "answer_count": user_data.get("answerCount", 0),
        "articles_count": user_data.get("articlesCount", 0),
        "pins_count": user_data.get("pinsCount", 0),
        "question_count": user_data.get("questionCount", 0),
        "voteup_count": user_data.get("voteupCount", 0),
        "thanked_count": user_data.get("thankedCount", 0),
        "description": user_data.get("description", ""),
    }


def get_my_url_token(session: requests.Session) -> str | None:
    """Get the logged-in user's url_token from /api/v4/me."""
    try:
        resp = session.get("https://www.zhihu.com/api/v4/me")
        if resp.status_code == 200:
            me = resp.json()
            return me.get("url_token") or me.get("urlToken")
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Graph construction
# ═══════════════════════════════════════════════════════════════════════════════


def build_social_graph(
    session: requests.Session,
    url_token: str,
    max_followees: int | None = None,
    max_followers: int | None = None,
    depth: int = 1,
    max_expand: int = 20,
    max_per_node: int = 50,
) -> nx.DiGraph:
    """Build a directed ego-network graph from Zhihu following/follower data.

    Args:
        session: Authenticated Session.
        url_token: The user's url_token to analyze.
        max_followees: Max number of followees to fetch (None = until exhausted).
        max_followers: Max number of followers to fetch (None = until exhausted).
        depth: Graph depth.  1 = ego-network (followees + followers only).
            Higher values recursively expand followees at each hop level.
        max_expand: Max nodes to expand *per hop level* (default 20).
            Mutuals are expanded first, then nodes sorted by follower count.
        max_per_node: Max followees fetched per expanded node (default 50).

    Returns:
        A ``nx.DiGraph`` where each node has ``name``, ``follower_count``,
        ``is_central``, ``hop`` (0 = central, 1 = direct, ≥2 = deeper),
        and optionally ``is_mutual`` attributes.  Edges have a ``relation``
        attribute set to ``"followee"`` or ``"follower"``.
    """
    G = nx.DiGraph()

    # ── central user ──────────────────────────────────────────────────────
    print(f"Fetching profile for {url_token} …")
    profile = fetch_member_profile(session, url_token)
    if profile:
        central_name: str = profile.get("name", url_token)
        G.add_node(
            url_token,
            name=central_name,
            follower_count=profile.get("follower_count", 0),
            following_count=profile.get("following_count", 0),
            is_central=True,
            is_mutual=False,
            hop=0,
        )
        print(
            f"  {central_name}  关注: {profile.get('following_count', 0)}  关注者: {profile.get('follower_count', 0)}"
        )
    else:
        print("  ⚠  Could not fetch profile, using url_token as label.")
        G.add_node(
            url_token,
            name=url_token,
            follower_count=0,
            following_count=0,
            is_central=True,
            is_mutual=False,
            hop=0,
        )
        central_name = url_token

    # ── followees (关注的用户) ─────────────────────────────────────────────
    print(f"Fetching followees (max: {max_followees or 'all'}) …")
    followees = fetch_followees(session, url_token, max_items=max_followees)
    followee_tokens: set[str] = set()
    for item in followees:
        token: str = item.get("url_token", "")
        if not token:
            continue
        followee_tokens.add(token)
        if token not in G:
            G.add_node(
                token,
                name=item.get("name", token),
                follower_count=item.get("follower_count", 0),
                headline=item.get("headline", ""),
                is_central=False,
                is_mutual=False,
                hop=1,
            )
        G.add_edge(url_token, token, relation="followee")
    print(f"  {len(followees)} followees fetched")

    # ── followers (关注者 / 粉丝) ──────────────────────────────────────────
    print(f"Fetching followers (max: {max_followers or 'all'}) …")
    followers = fetch_followers(session, url_token, max_items=max_followers)
    follower_tokens: set[str] = set()
    for item in followers:
        token = item.get("url_token", "")
        if not token:
            continue
        follower_tokens.add(token)
        if token not in G:
            G.add_node(
                token,
                name=item.get("name", token),
                follower_count=item.get("follower_count", 0),
                headline=item.get("headline", ""),
                is_central=False,
                is_mutual=False,
                hop=1,
            )
        G.add_edge(token, url_token, relation="follower")
    print(f"  {len(followers)} followers fetched")

    # ── mark mutual follows ───────────────────────────────────────────────
    mutual = followee_tokens & follower_tokens
    for token in mutual:
        G.nodes[token]["is_mutual"] = True
    if mutual:
        print(f"  ⇄ {len(mutual)} mutual follows detected")

    # ── recursive depth expansion ──────────────────────────────────────────
    for level in range(1, depth):
        _expand_level(G, session, url_token, level, max_expand, max_per_node)
        if not any(d.get("hop") == level + 1 for _, d in G.nodes(data=True)):
            print(f"  (no new nodes at hop {level + 1}, stopping expansion)")
            break

    return G


def _expand_level(
    G: nx.DiGraph,
    session: requests.Session,
    central_token: str,
    source_hop: int,
    max_expand: int,
    max_per_node: int,
) -> None:
    """Expand nodes at *source_hop* by fetching *their* followees."""
    target_hop = source_hop + 1

    candidates: list[tuple[int, int, str]] = []
    for n, d in G.nodes(data=True):
        if d.get("hop") != source_hop:
            continue
        if source_hop == 1 and d.get("is_central", False):
            continue
        fc = d.get("follower_count", 0)
        is_mut = 1 if d.get("is_mutual") else 0
        candidates.append((is_mut, fc, n))
    candidates.sort(reverse=True)

    expand_tokens = [t for _, _, t in candidates[:max_expand]]
    if not expand_tokens:
        return

    print(
        f"\nHop {source_hop} → {target_hop}: expanding {len(expand_tokens)} nodes (max {max_per_node} followees each) …"
    )

    for i, token in enumerate(expand_tokens):
        name = G.nodes[token].get("name", token)
        print(f"  [{i + 1}/{len(expand_tokens)}] {name[:30]} …", end=" ", flush=True)
        try:
            items = fetch_followees(session, token, max_items=max_per_node)
        except Exception:
            print("✗ (fetch error)")
            continue

        added = 0
        for item in items:
            child_token: str = item.get("url_token", "")
            if not child_token or child_token == central_token:
                continue
            if child_token not in G:
                G.add_node(
                    child_token,
                    name=item.get("name", child_token),
                    follower_count=item.get("follower_count", 0),
                    headline=item.get("headline", ""),
                    is_central=False,
                    is_mutual=False,
                    hop=target_hop,
                )
            G.add_edge(token, child_token, relation="followee")
            added += 1
        print(f"{added} followees")


# ═══════════════════════════════════════════════════════════════════════════════
# Graph statistics
# ═══════════════════════════════════════════════════════════════════════════════


def _dump_json(G: nx.DiGraph) -> None:
    """Dump all collected user-follower data as JSON to stdout."""
    users: list[dict[str, Any]] = []
    for node, data in G.nodes(data=True):
        users.append({
            "url_token": node,
            "name": data.get("name", node),
            "follower_count": data.get("follower_count", 0),
            "following_count": data.get("following_count", 0),
            "headline": data.get("headline", ""),
            "is_central": data.get("is_central", False),
            "is_mutual": data.get("is_mutual", False),
            "hop": data.get("hop", 0),
        })
    # Sort by follower_count descending
    users.sort(key=lambda u: u["follower_count"], reverse=True)

    output = {
        "total_users": len(users),
        "users": users,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


def print_graph_stats(G: nx.DiGraph, top_n: int = 10) -> None:
    """Print a human-readable statistics summary for the social graph.

    Args:
        G: The social graph.
        top_n: Number of top-by-followers entries to show (default 10).
    """
    central_nodes = [n for n, d in G.nodes(data=True) if d.get("is_central")]
    central = central_nodes[0] if central_nodes else None

    mutual_nodes = [n for n, d in G.nodes(data=True) if d.get("is_mutual")]
    followee_only = [n for n in (G.successors(central) if central else []) if n not in mutual_nodes]
    follower_only = [n for n in (G.predecessors(central) if central else []) if n not in mutual_nodes]

    max_hop = max((d.get("hop", 0) for _, d in G.nodes(data=True)), default=0)
    hop_counts: dict[int, int] = {}
    for _, d in G.nodes(data=True):
        h = d.get("hop", 0)
        hop_counts[h] = hop_counts.get(h, 0) + 1

    print()
    print("=" * 60)
    print("  Social Graph Statistics")
    print("=" * 60)
    print(f"  Total nodes:     {G.number_of_nodes()}")
    print(f"  Total edges:     {G.number_of_edges()}")
    if central:
        name = G.nodes[central].get("name", central)
        print(f"  Central user:    {name}")
        print(f"    Followees:     {len(followee_only) + len(mutual_nodes)}")
        print(f"    Followers:     {len(follower_only) + len(mutual_nodes)}")
        print(f"    Mutual:        {len(mutual_nodes)}")
    if max_hop >= 2:
        for h in range(2, max_hop + 1):
            print(f"    Hop-{h} nodes:  {hop_counts.get(h, 0)}")
    print()

    # Shared followees (followed by ≥2 nodes at the same parent hop level)
    if max_hop >= 2:
        _incoming: dict[str, list[str]] = {}
        for u, v, d in G.edges(data=True):
            if d.get("relation") == "followee" and G.nodes[v].get("hop", 0) >= 2:
                _incoming.setdefault(v, []).append(u)
        shared = [(v, refs) for v, refs in _incoming.items() if len(refs) >= 2]
        if shared:
            shared.sort(key=lambda x: -len(x[1]))
            print(f"  Shared followees ({len(shared)}):")
            for v, refs in shared[:10]:
                vname = G.nodes[v].get("name", v)[:24]
                vhop = G.nodes[v].get("hop", "?")
                rnames = ", ".join(G.nodes[r].get("name", r)[:12] for r in refs[:4])
                if len(refs) > 4:
                    rnames += f" (+{len(refs) - 4})"
                print(f"    {vname:<26s} hop={vhop} ← {rnames}")
            if len(shared) > 10:
                print(f"    … and {len(shared) - 10} more")
            print()

        # Triangles / clustering
        try:
            tri = nx.triangles(G)
            tri_count = sum(tri.values()) // 3
            if tri_count > 0:
                tri_nodes = [(n, t) for n, t in tri.items() if t > 0]
                tri_nodes.sort(key=lambda x: -x[1])
                print(f"  Triangles:       {tri_count}")
                print(f"  Clustering coeff: {nx.average_clustering(G):.4f}")
                print("  Top triangle nodes:")
                for n, t in tri_nodes[:8]:
                    name = G.nodes[n].get("name", n)[:24]
                    hop = G.nodes[n].get("hop", "?")
                    print(f"    {name:<26s} hop={hop}  △={t}")
                print()
        except Exception:
            pass

    # Top nodes by follower count
    top = sorted(G.nodes(data=True), key=lambda x: x[1].get("follower_count", 0), reverse=True)[:top_n]
    print(f"  Top {len(top)} nodes by followers:")
    for node, data in top:
        name = data.get("name", node)
        fc = data.get("follower_count", 0)
        deg = G.degree(node)
        marks = ""
        if data.get("is_central"):
            marks += "★"
        if data.get("is_mutual"):
            marks += "⇄"
        hop = data.get("hop", 1)
        if hop >= 2:
            marks += f"ʰ{hop}"
        print(f"    {marks:<5s} {name[:20]:<20s}  followers={fc:<6d}  deg={deg}")

    # Mutual follows list
    if mutual_nodes:
        print(f"\n  Mutual follows ({len(mutual_nodes)}):")
        for n in mutual_nodes[:20]:
            name = G.nodes[n].get("name", n)
            print(f"    ⇄ {name}")
        if len(mutual_nodes) > 20:
            print(f"    … and {len(mutual_nodes) - 20} more")

    print("=" * 60)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

HOP_COLORS = [
    "#bdc3c7",  # hop 2 — light gray
    "#d5b8d6",  # hop 3 — lavender
    "#ffeaa7",  # hop 4 — pale yellow
    "#81ecec",  # hop 5 — pale cyan
]


def visualize_social_graph(
    G: nx.DiGraph,
    output_path: str = "",
    layout: str = "spring",
    width: int = 1600,
    height: int = 1200,
) -> None:
    """Render the social graph to a PNG file with Plotly.

    Args:
        G: The directed social graph.
        output_path: Where to save the image.  Defaults to
            ``plots/zhihu_social_graph.png``.
        layout: One of ``spring``, ``kamada_kawai``, ``circular``, ``shell``.
        width: Image width in pixels.
        height: Image height in pixels.
    """
    if not output_path:
        output_path = str(DATA_DIR / "plots" / "zhihu_social_graph.png")

    if G.number_of_nodes() == 0:
        print("Graph has no nodes to visualize.")
        return

    central_nodes = [n for n, d in G.nodes(data=True) if d.get("is_central")]
    central = central_nodes[0] if central_nodes else list(G.nodes())[0]
    central_name = G.nodes[central].get("name", central)

    mutual_set: set[str] = {n for n, d in G.nodes(data=True) if d.get("is_mutual")}
    followee_set: set[str] = set(G.successors(central)) - mutual_set
    follower_set: set[str] = set(G.predecessors(central)) - mutual_set

    cat_map: dict[str, int] = {}
    for n in G.nodes():
        if G.nodes[n].get("is_central"):
            cat_map[n] = 0
        elif G.nodes[n].get("is_mutual"):
            cat_map[n] = 1
        elif n in followee_set:
            cat_map[n] = 2
        elif n in follower_set:
            cat_map[n] = 3
        else:
            h = G.nodes[n].get("hop", 2)
            cat_map[n] = 3 + min(h - 1, len(HOP_COLORS))

    categories: list[tuple[str, str]] = [
        ("#e74c3c", "自己"),
        ("#2ecc71", "互相关注"),
        ("#3498db", "关注的用户"),
        ("#f39c12", "关注者"),
    ]
    used_hops = sorted({G.nodes[n].get("hop", 0) for n in G.nodes() if G.nodes[n].get("hop", 0) >= 2})
    for h in used_hops:
        color_idx = min(h - 2, len(HOP_COLORS) - 1)
        categories.append((HOP_COLORS[color_idx], f"hop {h}"))

    print(f"Computing {layout} layout for {G.number_of_nodes()} nodes …")

    if layout == "spring":
        pos = nx.spring_layout(G, k=4.0, iterations=100, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "shell":
        others = [n for n in G.nodes() if n != central]
        pos = nx.shell_layout(G, nlist=[[central], others])
    else:
        pos = nx.spring_layout(G, k=4.0, iterations=100, seed=42)

    def _node_size(node: str) -> float:
        if G.nodes[node].get("is_central"):
            return 28
        fc = G.nodes[node].get("follower_count", 0)
        return max(5, min(18, 5 + 4 * np.log1p(fc)))

    edge_traces: list[go.Scatter] = []

    followee_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "followee"]
    follower_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "follower"]

    def _make_edge_trace(edges: list[tuple[str, str]], color: str, name: str) -> go.Scatter | None:
        if not edges:
            return None
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        for u, v in edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        return go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 0.6, "color": color},
            opacity=0.3,
            hoverinfo="none",
            showlegend=True,
            name=name,
        )

    fe_trace = _make_edge_trace(followee_edges, "#3498db", "关注的用户 →")
    if fe_trace:
        edge_traces.append(fe_trace)

    fr_trace = _make_edge_trace(follower_edges, "#f39c12", "→ 关注者")
    if fr_trace:
        edge_traces.append(fr_trace)

    node_traces: list[go.Scatter] = []
    for ci, (color, label) in enumerate(categories):
        nodelist = [n for n, c in cat_map.items() if c == ci]
        if not nodelist:
            continue
        nx_vals = [pos[n][0] for n in nodelist]
        ny_vals = [pos[n][1] for n in nodelist]
        sizes = [_node_size(n) for n in nodelist]

        hover_texts: list[str] = []
        for n in nodelist:
            name = G.nodes[n].get("name", n)
            fc = G.nodes[n].get("follower_count", 0)
            headline = G.nodes[n].get("headline", "")
            parts = [f"<b>{name}</b>", f"关注者: {fc}"]
            if headline:
                parts.append(headline[:80])
            hover_texts.append("<br>".join(parts))

        if ci >= 4:
            node_traces.append(
                go.Scatter(
                    x=nx_vals,
                    y=ny_vals,
                    mode="markers",
                    marker={
                        "size": [s * 0.6 for s in sizes],
                        "color": color,
                        "opacity": 0.55,
                        "line": {"width": 0.4, "color": "#999999"},
                    },
                    hoverinfo="text",
                    hovertext=hover_texts,
                    name=f"{label} ({len(nodelist)})",
                )
            )
        else:
            node_traces.append(
                go.Scatter(
                    x=nx_vals,
                    y=ny_vals,
                    mode="markers+text",
                    marker={
                        "size": sizes,
                        "color": color,
                        "line": {"width": 0.8, "color": "#ffffff"},
                    },
                    text=[G.nodes[n].get("name", n) for n in nodelist],
                    textposition="middle center",
                    textfont={"size": 8, "color": "#222222"},
                    hoverinfo="text",
                    hovertext=hover_texts,
                    name=f"{label} ({len(nodelist)})",
                )
            )

    n_followees = len(followee_set) + len(mutual_set)
    n_followers = len(follower_set) + len(mutual_set)
    n_deep = sum(1 for _, d in G.nodes(data=True) if d.get("hop", 0) >= 2)

    title_text = (
        f"知乎关注网络社交图谱 — {central_name}<br>"
        f"<sup>"
        f"节点 {G.number_of_nodes()}  ·  边 {G.number_of_edges()}  ·  "
        f"关注 {n_followees}  ·  关注者 {n_followers}  ·  互关 {len(mutual_set)}"
    )
    if n_deep:
        title_text += f"  ·  深层 {n_deep}"
    title_text += "</sup>"

    fig = go.Figure(
        data=edge_traces + node_traces,
        layout=go.Layout(
            title={"text": title_text, "font": {"size": 18}},
            showlegend=True,
            legend={
                "x": 1.02,
                "y": 1,
                "xanchor": "left",
                "font": {"size": 11},
                "bgcolor": "rgba(255,255,255,0.85)",
                "bordercolor": "#cccccc",
                "borderwidth": 1,
            },
            hovermode="closest",
            margin={"b": 20, "l": 20, "r": 20, "t": 80},
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            template="plotly_white",
            width=width,
            height=height,
        ),
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(output_path, scale=2, width=width, height=height)
        print(f"Social graph saved to: {output_path}")
    except Exception:
        print("Tip: kaleido is not installed, unable to save PNG. Install with: pip install kaleido")

    fig.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_headers_from_curl(curl_text: str) -> dict[str, str]:
    """Extract Cookie and User-Agent from a browser cURL command.

    Accepts the full cURL text (as copied from Chrome DevTools → Copy as cURL).
    """
    headers: dict[str, str] = {}

    # Try to extract Cookie from -H 'cookie: ...' or -H 'Cookie: ...'
    cookie_match = re.search(r"-H\s+'cookie:\s*([^']+)'\s", curl_text, re.IGNORECASE)
    if not cookie_match:
        cookie_match = re.search(r'-H\s+"cookie:\s*([^"]+)"', curl_text, re.IGNORECASE)
    if not cookie_match:
        cookie_match = re.search(r"-H\s+'Cookie:\s*([^']+)'\s", curl_text)
    if not cookie_match:
        cookie_match = re.search(r'-H\s+"Cookie:\s*([^"]+)"', curl_text)
    if cookie_match:
        headers["Cookie"] = cookie_match.group(1)

    # Try to extract User-Agent
    ua_match = re.search(r"-H\s+'user-agent:\s*([^']+)'", curl_text, re.IGNORECASE)
    if not ua_match:
        ua_match = re.search(r'-H\s+"user-agent:\s*([^"]+)"', curl_text, re.IGNORECASE)
    if ua_match:
        headers["User-Agent"] = ua_match.group(1)

    return headers


def main(
    url_token: str | None = None,
    max_followees: int | None = None,
    max_followers: int | None = None,
    output: str = "",
    layout: str = "spring",
    no_viz: bool = False,
    json_output: bool = False,
    depth: int = 1,
    max_expand: int = 20,
    max_per_node: int = 50,
    top_n: int = 10,
    cookie: str = "",
    user_agent: str = "",
    curl_text: str = "",
) -> None:
    """Build and visualize a Zhihu social network graph.

    Authentication: provide either ``--cookie`` + ``--user-agent``, or
    ``--curl`` (paste the full cURL command from browser DevTools).
    """
    # Resolve authentication
    if curl_text:
        extracted = _extract_headers_from_curl(curl_text)
        cookie = cookie or extracted.get("Cookie", "")
        user_agent = user_agent or extracted.get("User-Agent", "")

    if not cookie:
        print(
            "Error: No cookie provided.  Use --cookie or --curl to authenticate.",
            file=sys.stderr,
        )
        sys.exit(1)

    session = requests.Session()
    session.headers.update({
        "Cookie": cookie,
        "User-Agent": user_agent or "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    })

    if not output:
        output = str(DATA_DIR / "plots" / "zhihu_social_graph.png")

    # Resolve url_token
    if not url_token:
        url_token = get_my_url_token(session)
        if not url_token:
            print(
                "Error: Cannot detect your url_token.  Pass --url-token explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Using logged-in user: {url_token}")

    # Build the graph
    G = build_social_graph(
        session,
        url_token,
        max_followees=max_followees,
        max_followers=max_followers,
        depth=depth,
        max_expand=max_expand,
        max_per_node=max_per_node,
    )

    if G.number_of_nodes() <= 1:
        print("Graph is empty (only the central node).  No followees or followers were returned.")
        return

    # ── JSON output mode ────────────────────────────────────────────────────
    if json_output:
        _dump_json(G)
        return

    # Stats
    print_graph_stats(G, top_n=top_n)

    # Visualize
    if not no_viz:
        visualize_social_graph(G, output_path=output, layout=layout)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zhihu social network graph visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With explicit cookie and user-agent
  python zhihu_social_graph.py --cookie "d_c0=...; z_c0=..." --user-agent "Mozilla/5.0 ..."

  # Paste browser cURL (easiest)
  python zhihu_social_graph.py --curl "$(pbpaste)"

  # Analyze another user (not yourself)
  python zhihu_social_graph.py --cookie "..." --url-token some-user

  # 2-level depth expansion
  python zhihu_social_graph.py --cookie "..." --depth 2 --max-expand 15
        """,
    )

    # Authentication
    auth = parser.add_argument_group("Authentication (required)")
    auth.add_argument(
        "--cookie",
        default="",
        help="Cookie string from browser (must include d_c0 and z_c0)",
    )
    auth.add_argument(
        "--user-agent",
        default="",
        help="User-Agent string (auto-detected from --curl if omitted)",
    )
    auth.add_argument(
        "--curl",
        default="",
        help="Full cURL command text (pasted from browser DevTools → Copy as cURL)",
    )

    # Target
    target = parser.add_argument_group("Target")
    target.add_argument(
        "--url-token",
        help="User url_token to analyze (auto-detects logged-in user if omitted)",
    )

    # Limits
    limits = parser.add_argument_group("Fetch limits")
    limits.add_argument(
        "--max-followees",
        type=int,
        default=None,
        help="Max followees to fetch (default: unlimited)",
    )
    limits.add_argument(
        "--max-followers",
        type=int,
        default=None,
        help="Max followers to fetch (default: unlimited)",
    )

    # Depth
    depth_group = parser.add_argument_group("Depth expansion")
    depth_group.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Graph depth: 1 = ego-network, ≥2 = recursively expand (default: 1)",
    )
    depth_group.add_argument(
        "--max-expand",
        type=int,
        default=20,
        help="Max nodes to expand per hop level (default: 20)",
    )
    depth_group.add_argument(
        "--max-per-node",
        type=int,
        default=50,
        help="Max followees fetched per expanded node (default: 50)",
    )

    # Output
    out = parser.add_argument_group("Output")
    out.add_argument(
        "--output", "-o",
        default=str(DATA_DIR / "plots" / "zhihu_social_graph.png"),
        help="Output image path",
    )
    out.add_argument(
        "--layout",
        choices=["spring", "kamada_kawai", "circular", "shell"],
        default="spring",
        help="Graph layout algorithm (default: spring)",
    )
    out.add_argument(
        "--no-viz",
        action="store_true",
        help="Print statistics only, skip image generation",
    )
    out.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Dump all collected user-follower data as JSON and exit (skips stats and visualization)",
    )
    out.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top-by-followers entries to show in stats output (default: 10, no effect with --json)",
    )

    args = parser.parse_args()
    main(
        url_token=args.url_token or None,
        max_followees=args.max_followees,
        max_followers=args.max_followers,
        output=args.output,
        layout=args.layout,
        no_viz=args.no_viz,
        json_output=args.json_output,
        depth=args.depth,
        max_expand=args.max_expand,
        max_per_node=args.max_per_node,
        top_n=args.top_n,
        cookie=args.cookie,
        user_agent=args.user_agent,
        curl_text=args.curl,
    )
