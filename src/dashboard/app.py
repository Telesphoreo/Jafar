"""
Jafar ML Dashboard.

Flask app for browsing ML data, running bot detection, and HITL labeling.
Connects to the same Postgres database as the pipeline using a sync engine.

Usage:
    uv run dashboard
    # or
    uv run flask --app src.dashboard.app run --debug
"""

import logging
import os
import math
import threading
from datetime import datetime

from flask import Flask, flash, redirect, render_template, request, url_for
from sqlalchemy import create_engine, func, select, desc, asc, case
from sqlalchemy.orm import Session, sessionmaker

from src.models import (
    AccountScore,
    Base,
    BlockedAccount,
    SignalJudgment,
    HumanLabel,
    PipelineRun,
    Tweet,
    WatchedAccount,
)

logger = logging.getLogger("jafar.dashboard")
PER_PAGE = 50


def get_db_url() -> str:
    """Get database URL from environment, converting to sync driver."""
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set. Copy .env.example to .env and configure it."
        )
    # Strip async driver — we use sync psycopg2
    url = url.replace("postgresql+asyncpg://", "postgresql://")
    if not url.startswith("postgresql://"):
        url = "postgresql://" + url.split("://", 1)[-1]
    return url


def create_app() -> Flask:
    """Create and configure the Flask app."""
    app = Flask(__name__, template_folder="templates")
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "jafar-dev-key-change-me")

    # Load .env if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Sync SQLAlchemy engine
    engine = create_engine(get_db_url(), pool_pre_ping=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(engine)

    # Track ML analysis state
    _ml_status = {
        "running": False,
        "task": None,        # "scoring" or "judging"
        "progress": "",      # e.g. "12/203 accounts scored"
        "last_error": None,
    }

    def get_session() -> Session:
        return SessionLocal()

    # Display name mappings
    CLUSTER_DISPLAY = {
        "high_signal": "high signal",
        "news_aggregator": "news aggregator",
        "low_activity": "low activity",
        "casual": "casual",
        "unclustered": "unclustered",
    }

    FEATURE_DISPLAY = {
        "avg_content_length": "avg content length",
        "url_ratio": "url ratio",
        "unique_content_ratio": "unique content",
        "question_ratio": "question ratio",
        "avg_likes": "avg likes",
        "avg_retweets": "avg retweets",
        "avg_replies": "avg replies",
        "reply_to_like_ratio": "reply to like ratio",
        "engagement_per_view": "engagement per view",
        "engagement_consistency": "engagement consistency",
        "trend_coverage": "trend coverage",
        "trend_coverage_ratio": "trend coverage ratio",
        "pipeline_run_appearances": "pipeline runs seen",
        "recurrence_ratio": "recurrence ratio",
        "active_hours_spread": "active hours spread",
        "tweet_frequency": "tweet frequency",
    }

    @app.template_filter("cluster_name")
    def cluster_name_filter(value):
        return CLUSTER_DISPLAY.get(value, value or "—")

    @app.template_filter("feature_name")
    def feature_name_filter(value):
        return FEATURE_DISPLAY.get(value, value.replace("_", " "))

    @app.template_filter("compact_number")
    def compact_number_filter(value):
        try:
            n = float(value)
        except (TypeError, ValueError):
            return value
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.1f}b"
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}m"
        if n >= 1_000:
            return f"{n / 1_000:.1f}k"
        return f"{n:.0f}"

    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------

    @app.route("/")
    def overview():
        with get_session() as s:
            total_tweets = s.execute(
                select(func.count()).select_from(Tweet)
            ).scalar_one()
            total_accounts = s.execute(
                select(func.count(func.distinct(Tweet.username)))
            ).scalar_one()
            total_labeled = s.execute(
                select(func.count()).select_from(HumanLabel)
            ).scalar_one()
            total_scored = s.execute(
                select(func.count()).select_from(AccountScore)
            ).scalar_one()
            total_runs = s.execute(
                select(func.count()).select_from(PipelineRun)
            ).scalar_one()

            # Anomaly count (IsolationForest-flagged)
            total_bot_suspects = s.execute(
                select(func.count()).select_from(AccountScore)
                .where(AccountScore.is_anomaly.is_(True))
            ).scalar_one()

            # Last scored time
            last_scored_row = s.execute(
                select(AccountScore.scored_at)
                .order_by(AccountScore.scored_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            last_scored = (
                last_scored_row.strftime("%Y-%m-%d %H:%M")
                if last_scored_row else None
            )

            # Recent runs
            recent_runs = s.execute(
                select(PipelineRun)
                .order_by(PipelineRun.run_id.desc())
                .limit(10)
            ).scalars().all()

            # Top accounts by tweet count
            top_q = (
                select(
                    Tweet.username,
                    func.count().label("tweet_count"),
                    func.avg(Tweet.likes).label("avg_likes"),
                )
                .group_by(Tweet.username)
                .order_by(desc("tweet_count"))
                .limit(15)
            )
            top_rows = s.execute(top_q).all()

            usernames = [r.username for r in top_rows]
            labels = {}
            if usernames:
                label_rows = s.execute(
                    select(HumanLabel).where(HumanLabel.username.in_(usernames))
                ).scalars().all()
                labels = {l.username: l.label for l in label_rows}

            top_accounts = [
                {
                    "username": r.username,
                    "tweet_count": r.tweet_count,
                    "avg_likes": r.avg_likes or 0,
                    "label": labels.get(r.username),
                }
                for r in top_rows
            ]

            label_counts = {}
            lc_rows = s.execute(
                select(HumanLabel.label, func.count())
                .group_by(HumanLabel.label)
            ).all()
            label_counts = {r[0]: r[1] for r in lc_rows}

            # Judge stats
            total_judged = s.execute(
                select(func.count(func.distinct(SignalJudgment.username)))
            ).scalar_one()
            total_blocked = s.execute(
                select(func.count()).select_from(BlockedAccount)
            ).scalar_one()
            total_unjudged = max(0, total_scored - total_judged)

        return render_template(
            "overview.html",
            total_tweets=total_tweets,
            total_accounts=total_accounts,
            total_labeled=total_labeled,
            total_scored=total_scored,
            total_bot_suspects=total_bot_suspects,
            total_runs=total_runs,
            last_scored=last_scored,
            recent_runs=recent_runs,
            top_accounts=top_accounts,
            label_counts=label_counts,
            total_judged=total_judged,
            total_blocked=total_blocked,
            total_unjudged=total_unjudged,
        )

    # ------------------------------------------------------------------
    # ML analysis (run bot detection + signal scoring)
    # ------------------------------------------------------------------

    @app.route("/api/ml-status")
    def ml_status():
        from flask import jsonify
        return jsonify(_ml_status)

    @app.route("/api/run-ml", methods=["POST"])
    def run_ml():
        if _ml_status["running"]:
            flash("ml analysis is already running.", "error")
            return redirect(url_for("overview"))

        _ml_status["running"] = True
        _ml_status["task"] = "scoring"
        _ml_status["progress"] = "starting..."
        _ml_status["last_error"] = None

        def _run_analysis():
            """Run ML analysis in a background thread."""
            import asyncio
            asyncio.run(_run_ml_async())

        async def _run_ml_async():
            try:
                from src.database import init_db, close_db
                await init_db(os.environ.get("DATABASE_URL", ""))

                from src.ml import BotScorer, AccountScorer

                # 1. Bot detection
                _ml_status["progress"] = "training anomaly detector..."
                bot_scorer = BotScorer()
                training_stats = await bot_scorer.train(min_tweets_per_account=5)
                eligible = training_stats.get('accounts_eligible', 0)
                logger.info(f"Bot detection trained on {eligible} accounts")

                bot_scores = {}
                if bot_scorer.model is not None:
                    _ml_status["progress"] = f"scoring {eligible} accounts for anomalies..."
                    all_scores = await bot_scorer.score_all_accounts(min_tweets=5)
                    bot_scores = {s["username"]: s for s in all_scores}

                # 2. Signal scoring
                _ml_status["progress"] = "computing signal scores..."
                account_scorer = AccountScorer()
                all_accounts = await account_scorer.analyze(min_tweets_per_account=3)

                # 3. Persist to account_scores table
                _ml_status["progress"] = f"saving {len(all_accounts)} scores to db..."
                now = datetime.now()
                with get_session() as s:
                    for acct in all_accounts:
                        username = acct["username"]
                        bot_info = bot_scores.get(username, {})

                        existing = s.execute(
                            select(AccountScore)
                            .where(AccountScore.username == username)
                        ).scalar_one_or_none()

                        score_data = dict(
                            garbage_score=bot_info.get("garbage_score", 0.0),
                            is_anomaly=bot_info.get("anomaly", False),
                            signal_score=acct.get("signal_score", 0.0),
                            cluster_label=acct.get("cluster_label"),
                            tweet_count=acct.get("tweet_count", 0),
                            features_json=acct.get("features"),
                            scored_at=now,
                        )
                        if existing:
                            for k, v in score_data.items():
                                setattr(existing, k, v)
                        else:
                            s.add(AccountScore(username=username, **score_data))
                    s.commit()

                anomaly_count = sum(1 for b in bot_scores.values() if b.get("anomaly"))
                logger.info(
                    f"ML analysis complete: {len(all_accounts)} accounts scored, "
                    f"{anomaly_count} anomalies detected by IsolationForest"
                )

            except Exception as e:
                logger.error(f"ML analysis failed: {e}", exc_info=True)
                _ml_status["last_error"] = str(e)
                _ml_status["progress"] = f"failed: {e}"
            finally:
                try:
                    await close_db()
                except Exception:
                    pass
                _ml_status["running"] = False
                _ml_status["task"] = None
                if not _ml_status["last_error"]:
                    _ml_status["progress"] = ""

        thread = threading.Thread(target=_run_analysis, daemon=True)
        thread.start()

        flash("scoring started in the background.", "info")
        return redirect(url_for("overview"))

    # ------------------------------------------------------------------
    # LLM Judge (run Gemini bot evaluation on scored accounts)
    # ------------------------------------------------------------------

    @app.route("/api/run-judge", methods=["POST"])
    def run_judge():
        if _ml_status["running"]:
            flash("an ml task is already running.", "error")
            return redirect(url_for("overview"))

        _ml_status["running"] = True
        _ml_status["task"] = "judging"
        _ml_status["progress"] = "starting..."
        _ml_status["last_error"] = None

        def _run_judge_thread():
            import asyncio
            asyncio.run(_run_judge_async())

        async def _run_judge_async():
            import asyncio
            try:
                from src.database import init_db, close_db, get_session as async_get_session
                await init_db(os.environ.get("DATABASE_URL", ""))

                from src.ml.llm_judge import BotJudge
                from src.models import AccountScore as AS, SignalJudgment as SJ, Tweet as TW
                from sqlalchemy import select as asel

                from src.config import config
                api_key = config.google.api_key
                model = config.google.model
                if not api_key:
                    raise RuntimeError("GOOGLE_API_KEY not set")

                judge = BotJudge(api_key=api_key, model=model)

                # Find accounts to judge: scored but not yet judged
                session = await async_get_session()
                try:
                    # Get already-judged usernames
                    judged_q = asel(SJ.username).distinct()
                    judged_rows = (await session.execute(judged_q)).scalars().all()
                    judged_set = set(judged_rows)

                    # Get scored accounts, prioritize anomalies
                    scored_q = (
                        asel(AS)
                        .order_by(AS.is_anomaly.desc(), AS.garbage_score.desc())
                    )
                    scored = (await session.execute(scored_q)).scalars().all()
                finally:
                    await session.close()

                # Filter to unjudged, cap at 30 per run (API cost control)
                to_judge = [s for s in scored if s.username not in judged_set]

                if not to_judge:
                    logger.info("No unjudged accounts to evaluate")
                    _ml_status["progress"] = "nothing to judge"
                    _ml_status["running"] = False
                    _ml_status["task"] = None
                    return

                total_to_judge = len(to_judge)
                logger.info(f"LLM judge: evaluating {total_to_judge} accounts...")
                _ml_status["progress"] = f"0/{total_to_judge} judged"
                judged_count = 0
                rate_limit_retries = 0
                max_retries = 3

                for acct in to_judge:
                    # Fetch tweets for this account
                    session = await async_get_session()
                    try:
                        tweet_rows = (await session.execute(
                            asel(TW)
                            .where(TW.username == acct.username)
                            .order_by(TW.created_at.desc())
                            .limit(20)
                        )).scalars().all()
                    finally:
                        await session.close()

                    tweets = [
                        {
                            "content": t.content,
                            "created_at": str(t.created_at) if t.created_at else "unknown",
                            "likes": t.likes,
                            "retweets": t.retweets,
                            "replies": t.replies,
                            "views": t.views,
                        }
                        for t in tweet_rows
                    ]

                    if not tweets:
                        continue

                    # Judge with rate limit retry
                    judgment = None
                    for attempt in range(max_retries + 1):
                        try:
                            judgment = await judge.judge_account(
                                username=acct.username,
                                tweets=tweets,
                                ml_features=acct.features_json,
                            )
                            break
                        except Exception as e:
                            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                                rate_limit_retries += 1
                                if attempt < max_retries:
                                    wait = 35 * (attempt + 1)
                                    _ml_status["progress"] = f"{judged_count}/{total_to_judge} judged / rate limited, retrying in {wait}s..."
                                    logger.warning(
                                        f"  Rate limited, waiting {wait}s "
                                        f"(attempt {attempt + 1}/{max_retries})..."
                                    )
                                    await asyncio.sleep(wait)
                                else:
                                    logger.warning(
                                        f"  Rate limit persists after {max_retries} retries. "
                                        f"Stopping. {judged_count} accounts judged so far."
                                    )
                                    break
                            else:
                                logger.error(f"  Failed to judge @{acct.username}: {e}")
                                break

                    if judgment is None:
                        if rate_limit_retries > max_retries:
                            break  # Stop the whole loop on persistent rate limits
                        continue

                    # Store judgment immediately (partial progress)
                    session = await async_get_session()
                    async with session.begin():
                        session.add(SJ(
                            username=acct.username,
                            garbage_probability=judgment["garbage_probability"],
                            confidence=judgment["confidence"],
                            classification=judgment["classification"],
                            reasoning=judgment["reasoning"],
                            signals=judgment["signals"],
                            ml_garbage_score=acct.garbage_score,
                        ))
                    await session.close()
                    judged_count += 1
                    classification = judgment['classification'].replace('_', ' ')
                    _ml_status["progress"] = f"{judged_count}/{total_to_judge} judged / @{acct.username}: {classification}"

                    logger.info(
                        f"  @{acct.username}: {judgment['classification']} "
                        f"(conf: {judgment['confidence']:.2f})"
                    )

                    # Auto-actions for high-confidence judgments
                    if judgment["confidence"] >= 0.8:
                        with get_session() as s:
                            if judgment["classification"] in ("garbage", "likely_garbage"):
                                # Auto-block garbage
                                exists = s.execute(
                                    select(BlockedAccount)
                                    .where(BlockedAccount.username == acct.username)
                                ).scalar_one_or_none()
                                if not exists:
                                    s.add(BlockedAccount(
                                        username=acct.username,
                                        reason=f"LLM judge: {judgment['classification']} "
                                               f"(conf {judgment['confidence']:.2f})",
                                    ))
                                    logger.info(f"    Auto-blocked @{acct.username}")

                            if judgment["classification"] in ("signal", "likely_signal"):
                                # Auto-label as signal
                                existing_label = s.execute(
                                    select(HumanLabel)
                                    .where(HumanLabel.username == acct.username)
                                ).scalar_one_or_none()
                                if not existing_label:
                                    s.add(HumanLabel(
                                        username=acct.username,
                                        label="signal",
                                        notes=f"LLM judge: {judgment['classification']} "
                                              f"(conf {judgment['confidence']:.2f})",
                                    ))
                                    logger.info(f"    Auto-labeled @{acct.username} as signal")

                            s.commit()

                    # Small delay between calls to avoid hitting rate limits
                    await asyncio.sleep(2)

                logger.info(f"LLM judge complete: {judged_count}/{len(to_judge)} accounts judged")

                # Auto-apply any pending actions from this and previous runs
                _ml_status["progress"] = "applying pending actions..."
                _apply_pending_actions(get_session)

            except Exception as e:
                logger.error(f"LLM judge failed: {e}", exc_info=True)
                _ml_status["last_error"] = str(e)
                _ml_status["progress"] = f"failed: {e}"
            finally:
                try:
                    await close_db()
                except Exception:
                    pass
                _ml_status["running"] = False
                _ml_status["task"] = None
                if not _ml_status["last_error"]:
                    _ml_status["progress"] = ""

        thread = threading.Thread(target=_run_judge_thread, daemon=True)
        thread.start()

        flash("llm judge started in the background. evaluating unjudged accounts.", "info")
        return redirect(url_for("overview"))

    # ------------------------------------------------------------------
    # Apply pending auto-actions from existing judgments
    # ------------------------------------------------------------------

    def _apply_pending_actions(session_factory):
        """Apply auto-block/auto-label to all high-confidence judgments."""
        with session_factory() as s:
            judgments = s.execute(
                select(SignalJudgment)
                .where(SignalJudgment.confidence >= 0.8)
            ).scalars().all()

            blocked = 0
            labeled = 0

            for j in judgments:
                if j.classification in ("garbage", "likely_garbage"):
                    exists = s.execute(
                        select(BlockedAccount)
                        .where(BlockedAccount.username == j.username)
                    ).scalar_one_or_none()
                    if not exists:
                        s.add(BlockedAccount(
                            username=j.username,
                            reason=f"llm judge: {j.classification} (conf {j.confidence:.2f})",
                        ))
                        blocked += 1

                elif j.classification in ("signal", "likely_signal"):
                    exists = s.execute(
                        select(HumanLabel)
                        .where(HumanLabel.username == j.username)
                    ).scalar_one_or_none()
                    if not exists:
                        s.add(HumanLabel(
                            username=j.username,
                            label="signal",
                            notes=f"llm judge: {j.classification} (conf {j.confidence:.2f})",
                        ))
                        labeled += 1

            s.commit()
            return blocked, labeled

    @app.route("/api/apply-judgments", methods=["POST"])
    def apply_judgments():
        blocked, labeled = _apply_pending_actions(get_session)
        flash(f"{blocked} accounts blocked, {labeled} accounts labeled as signal.", "success")
        return redirect(url_for("overview"))

    # ------------------------------------------------------------------
    # Accounts list
    # ------------------------------------------------------------------

    @app.route("/accounts")
    def accounts():
        page = request.args.get("page", 1, type=int)
        sort = request.args.get("sort", "tweet_count")
        order = request.args.get("order", "desc")
        filter_ = request.args.get("filter", "all")

        allowed_sorts = {
            "tweet_count", "avg_likes", "avg_retweets",
            "total_views", "username", "garbage_score", "signal_score",
        }
        if sort not in allowed_sorts:
            sort = "tweet_count"

        with get_session() as s:
            # Aggregate tweet stats per account, left-joined with scores and labels
            acct_stats = (
                select(
                    Tweet.username,
                    func.count().label("tweet_count"),
                    func.avg(Tweet.likes).label("avg_likes"),
                    func.avg(Tweet.retweets).label("avg_retweets"),
                    func.sum(Tweet.views).label("total_views"),
                )
                .group_by(Tweet.username)
                .subquery()
            )

            q = (
                select(
                    acct_stats.c.username,
                    acct_stats.c.tweet_count,
                    acct_stats.c.avg_likes,
                    acct_stats.c.avg_retweets,
                    acct_stats.c.total_views,
                    AccountScore.garbage_score,
                    AccountScore.signal_score,
                    AccountScore.cluster_label,
                    HumanLabel.label,
                    HumanLabel.notes.label("label_notes"),
                )
                .outerjoin(AccountScore, acct_stats.c.username == AccountScore.username)
                .outerjoin(HumanLabel, acct_stats.c.username == HumanLabel.username)
            )

            # Filter
            if filter_ == "unlabeled":
                q = q.where(HumanLabel.label.is_(None))
            elif filter_ == "garbage":
                q = q.where(HumanLabel.label == "garbage")
            elif filter_ == "signal":
                q = q.where(HumanLabel.label == "signal")
            elif filter_ == "unsure":
                q = q.where(HumanLabel.label == "unsure")
            elif filter_ == "suspects":
                q = q.where(AccountScore.is_anomaly.is_(True))

            # Sort
            sort_map = {
                "tweet_count": acct_stats.c.tweet_count,
                "avg_likes": acct_stats.c.avg_likes,
                "avg_retweets": acct_stats.c.avg_retweets,
                "total_views": acct_stats.c.total_views,
                "username": acct_stats.c.username,
                "garbage_score": AccountScore.garbage_score,
                "signal_score": AccountScore.signal_score,
            }
            sort_col = sort_map.get(sort, acct_stats.c.tweet_count)
            if order == "asc":
                q = q.order_by(asc(sort_col).nullslast())
            else:
                q = q.order_by(desc(sort_col).nullslast())

            total = s.execute(
                select(func.count()).select_from(q.subquery())
            ).scalar_one()
            total_pages = max(1, math.ceil(total / PER_PAGE))

            rows = s.execute(q.offset((page - 1) * PER_PAGE).limit(PER_PAGE)).all()

            account_list = [
                {
                    "username": r.username,
                    "tweet_count": r.tweet_count,
                    "avg_likes": r.avg_likes or 0,
                    "avg_retweets": r.avg_retweets or 0,
                    "total_views": r.total_views or 0,
                    "garbage_score": r.garbage_score,
                    "signal_score": r.signal_score,
                    "cluster_label": r.cluster_label,
                    "label": r.label,
                    "label_source": "LLM" if r.label_notes and "LLM judge" in r.label_notes else "HITL" if r.label else None,
                }
                for r in rows
            ]

        return render_template(
            "accounts.html",
            accounts=account_list,
            page=page,
            total=total,
            total_pages=total_pages,
            sort=sort,
            order=order,
            filter=filter_,
        )

    # ------------------------------------------------------------------
    # Account detail
    # ------------------------------------------------------------------

    @app.route("/account/<username>")
    def account_detail(username):
        with get_session() as s:
            stats_row = s.execute(
                select(
                    func.count().label("tweet_count"),
                    func.avg(Tweet.likes).label("avg_likes"),
                    func.avg(Tweet.retweets).label("avg_retweets"),
                    func.sum(Tweet.views).label("total_views"),
                )
                .where(Tweet.username == username)
            ).one()
            stats = {
                "tweet_count": stats_row.tweet_count,
                "avg_likes": stats_row.avg_likes or 0,
                "avg_retweets": stats_row.avg_retweets or 0,
                "total_views": stats_row.total_views or 0,
            }

            tweets = s.execute(
                select(Tweet)
                .where(Tweet.username == username)
                .order_by(Tweet.created_at.desc())
                .limit(100)
            ).scalars().all()

            judgment = s.execute(
                select(SignalJudgment)
                .where(SignalJudgment.username == username)
                .order_by(SignalJudgment.judged_at.desc())
                .limit(1)
            ).scalar_one_or_none()

            current_label = s.execute(
                select(HumanLabel)
                .where(HumanLabel.username == username)
            ).scalar_one_or_none()

            score = s.execute(
                select(AccountScore)
                .where(AccountScore.username == username)
            ).scalar_one_or_none()

            is_watched = s.execute(
                select(WatchedAccount)
                .where(WatchedAccount.username == username)
            ).scalar_one_or_none()

            is_blocked = s.execute(
                select(BlockedAccount)
                .where(BlockedAccount.username == username)
            ).scalar_one_or_none()

        return render_template(
            "account.html",
            username=username,
            stats=stats,
            tweets=tweets,
            judgment=judgment,
            current_label=current_label,
            score=score,
            is_watched=is_watched,
            is_blocked=is_blocked,
        )

    # ------------------------------------------------------------------
    # HITL review queue
    # ------------------------------------------------------------------

    @app.route("/review")
    def review():
        page = request.args.get("page", 1, type=int)
        filter_ = request.args.get("filter", "needs_review")

        with get_session() as s:
            # Blocked and labeled usernames — these are resolved
            blocked_sq = select(BlockedAccount.username).subquery()
            labeled_sq = select(HumanLabel.username).subquery()

            # Aggregate tweet stats
            acct_stats = (
                select(
                    Tweet.username,
                    func.count().label("tweet_count"),
                    func.avg(Tweet.likes).label("avg_likes"),
                )
                .group_by(Tweet.username)
                .having(func.count() >= 3)
                .subquery()
            )

            # Base query: judged accounts with their scores
            q = (
                select(
                    acct_stats.c.username,
                    acct_stats.c.tweet_count,
                    acct_stats.c.avg_likes,
                    AccountScore.garbage_score,
                    AccountScore.signal_score,
                    AccountScore.cluster_label,
                    SignalJudgment.classification.label("llm_classification"),
                    SignalJudgment.confidence.label("llm_confidence"),
                    SignalJudgment.reasoning.label("llm_reasoning"),
                    HumanLabel.label,
                )
                .join(SignalJudgment, acct_stats.c.username == SignalJudgment.username)
                .outerjoin(AccountScore, acct_stats.c.username == AccountScore.username)
                .outerjoin(HumanLabel, acct_stats.c.username == HumanLabel.username)
            )

            if filter_ == "needs_review":
                # Judged but not blocked and not labeled — the actual backlog
                # Also include "unsure" accounts that have new tweets since labeling
                from sqlalchemy import or_
                unsure_with_new_tweets = select(
                    HumanLabel.username
                ).join(
                    Tweet, Tweet.username == HumanLabel.username
                ).where(
                    HumanLabel.label == "unsure",
                    Tweet.scraped_at > HumanLabel.labeled_at,
                ).group_by(HumanLabel.username).subquery()

                q = q.where(
                    acct_stats.c.username.notin_(blocked_sq),
                    or_(
                        HumanLabel.label.is_(None),
                        acct_stats.c.username.in_(select(unsure_with_new_tweets)),
                    ),
                )
            elif filter_ == "unsure":
                q = q.where(HumanLabel.label == "unsure")
            elif filter_ == "resolved":
                # Already handled (blocked or labeled)
                q = q  # show all judged, including resolved
            # "all" shows everything judged

            # Sort: low confidence first (hardest calls), then by garbage score
            q = q.order_by(
                asc(SignalJudgment.confidence),
                desc(AccountScore.garbage_score).nullslast(),
            )

            total = s.execute(
                select(func.count()).select_from(q.subquery())
            ).scalar_one()

            # Counts for tabs
            needs_review_q = (
                select(func.count(func.distinct(SignalJudgment.username)))
                .where(
                    SignalJudgment.username.notin_(blocked_sq),
                    SignalJudgment.username.notin_(labeled_sq),
                )
            )
            count_needs_review = s.execute(needs_review_q).scalar_one()
            count_unsure = s.execute(
                select(func.count()).select_from(HumanLabel)
                .where(HumanLabel.label == "unsure")
            ).scalar_one()
            count_all_judged = s.execute(
                select(func.count(func.distinct(SignalJudgment.username)))
            ).scalar_one()

            counts = {
                "needs_review": count_needs_review,
                "unsure": count_unsure,
                "all": count_all_judged,
            }
            total_pages = max(1, math.ceil(total / 20))

            rows = s.execute(q.offset((page - 1) * 20).limit(20)).all()

            queue = []
            for r in rows:
                sample_tweets = s.execute(
                    select(Tweet)
                    .where(Tweet.username == r.username)
                    .order_by(Tweet.likes.desc())
                    .limit(4)
                ).scalars().all()

                queue.append({
                    "username": r.username,
                    "tweet_count": r.tweet_count,
                    "avg_likes": r.avg_likes or 0,
                    "garbage_score": r.garbage_score,
                    "signal_score": r.signal_score,
                    "cluster_label": r.cluster_label,
                    "llm_classification": r.llm_classification,
                    "llm_confidence": r.llm_confidence,
                    "llm_reasoning": r.llm_reasoning,
                    "label": r.label,
                    "sample_tweets": sample_tweets,
                })

        return render_template(
            "review.html",
            queue=queue,
            page=page,
            total=total,
            total_pages=total_pages,
            filter=filter_,
            counts=counts,
        )

    # ------------------------------------------------------------------
    # Pipeline runs
    # ------------------------------------------------------------------

    @app.route("/runs")
    def runs():
        with get_session() as s:
            all_runs = s.execute(
                select(PipelineRun).order_by(PipelineRun.run_id.desc())
            ).scalars().all()

        return render_template("runs.html", runs=all_runs)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @app.route("/search")
    def search():
        q = request.args.get("q", "").strip().lstrip("@")
        if not q:
            return redirect(url_for("accounts"))

        # Exact match — go straight to account page
        with get_session() as s:
            exact = s.execute(
                select(Tweet.username)
                .where(Tweet.username == q)
                .limit(1)
            ).scalar_one_or_none()

            if exact:
                return redirect(url_for("account_detail", username=exact))

            # Partial match — find accounts containing the query
            matches = s.execute(
                select(Tweet.username, func.count().label("tweet_count"))
                .where(Tweet.username.ilike(f"%{q}%"))
                .group_by(Tweet.username)
                .order_by(desc("tweet_count"))
                .limit(20)
            ).all()

        if len(matches) == 1:
            return redirect(url_for("account_detail", username=matches[0].username))

        return render_template("search.html", query=q, results=matches)

    # ------------------------------------------------------------------
    # HITL labeling
    # ------------------------------------------------------------------

    @app.route("/api/label", methods=["POST"])
    def label_account():
        username = request.form.get("username", "").strip()
        label = request.form.get("label", "").strip()
        notes = request.form.get("notes", "").strip()
        redirect_url = request.form.get("redirect", url_for("review"))

        valid_labels = {"signal", "garbage", "unsure"}
        if not username or label not in valid_labels:
            flash("invalid label request", "error")
            return redirect(redirect_url)

        with get_session() as s:
            existing = s.execute(
                select(HumanLabel).where(HumanLabel.username == username)
            ).scalar_one_or_none()

            if existing:
                existing.label = label
                existing.notes = notes or existing.notes
                existing.labeled_at = datetime.now()
            else:
                s.add(HumanLabel(
                    username=username,
                    label=label,
                    notes=notes or None,
                ))
            s.commit()

        flash(f"@{username} labeled as {label}.", "success")
        return redirect(redirect_url)

    # ------------------------------------------------------------------
    # Watch / Block account management
    # ------------------------------------------------------------------

    @app.route("/api/watch", methods=["POST"])
    def watch_account():
        username = request.form.get("username", "").strip()
        reason = request.form.get("reason", "").strip()
        redirect_url = request.form.get("redirect", url_for("overview"))

        if not username:
            flash("username required.", "error")
            return redirect(redirect_url)

        with get_session() as s:
            existing = s.execute(
                select(WatchedAccount).where(WatchedAccount.username == username)
            ).scalar_one_or_none()

            if existing:
                flash(f"@{username} is already watched.", "info")
            else:
                s.add(WatchedAccount(username=username, reason=reason or None))
                s.commit()
                flash(f"@{username} added to watch list.", "success")

        return redirect(redirect_url)

    @app.route("/api/unwatch", methods=["POST"])
    def unwatch_account():
        username = request.form.get("username", "").strip()
        redirect_url = request.form.get("redirect", url_for("overview"))

        with get_session() as s:
            existing = s.execute(
                select(WatchedAccount).where(WatchedAccount.username == username)
            ).scalar_one_or_none()
            if existing:
                s.delete(existing)
                s.commit()
                flash(f"@{username} removed from watch list.", "success")

        return redirect(redirect_url)

    @app.route("/api/block", methods=["POST"])
    def block_account():
        username = request.form.get("username", "").strip()
        reason = request.form.get("reason", "").strip()
        redirect_url = request.form.get("redirect", url_for("overview"))

        if not username:
            flash("username required.", "error")
            return redirect(redirect_url)

        with get_session() as s:
            existing = s.execute(
                select(BlockedAccount).where(BlockedAccount.username == username)
            ).scalar_one_or_none()

            if existing:
                flash(f"@{username} is already blocked.", "info")
            else:
                s.add(BlockedAccount(username=username, reason=reason or None))
                s.commit()
                flash(f"@{username} blocked.", "success")

        return redirect(redirect_url)

    @app.route("/api/unblock", methods=["POST"])
    def unblock_account():
        username = request.form.get("username", "").strip()
        redirect_url = request.form.get("redirect", url_for("overview"))

        with get_session() as s:
            existing = s.execute(
                select(BlockedAccount).where(BlockedAccount.username == username)
            ).scalar_one_or_none()
            if existing:
                s.delete(existing)
                s.commit()
                flash(f"@{username} unblocked.", "success")

        return redirect(redirect_url)

    return app


def main():
    """Entry point for `uv run dashboard`."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    app = create_app()
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    main()
