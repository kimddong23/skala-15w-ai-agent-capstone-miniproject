/**
 * 대상 사용자 선택 드롭다운 — 네이티브 `<select>` 를 대체하는 WAI-ARIA 선택 전용 combobox.
 *
 * 왜 직접 만드나: 네이티브 select 의 목록은 OS 마다 위치·모양이 달라 버튼 아래로 일정하게 펼쳐지지 않고, 한 줄에 ID·기간·
 *   기록 비율을 열로 정렬해 보여줄 수도 없다. 목록은 버튼 바로 아래(+4px)에 버튼과 같은 폭으로 열린다.
 * 책임: 열기/닫기, 방향키·Home/End·Enter·Esc, 바깥 클릭 닫기, 선택 값 통지. 사용자 목록·선택 상태는 부모가 소유한다.
 * 접근성: 포커스는 버튼에 머물고 `aria-activedescendant` 가 활성 옵션을 가리킨다(선택 전용 combobox 패턴).
 * E2E: 버튼 `data-testid="user-select"`(role combobox), 옵션 `data-testid="user-option"`(role option, `data-user-id`).
 *   네이티브 select 가 아니므로 `selectOption` 대신 버튼을 클릭한 뒤 옵션을 클릭한다.
 */
import { useEffect, useId, useRef, useState } from "react";
import { ChevronDown } from "react-bootstrap-icons";
import type { UserInfo } from "./events";
import { participantNo } from "./labels";
import { fmtDate, tsToMs } from "./time";

interface Props {
  users: UserInfo[];
  /** 선택된 사용자 ID. 목록이 비었거나 아직 없으면 null. */
  value: string | null;
  onChange: (id: string) => void;
}

// 기간은 연도를 한 번만 적는다("2016-04-13 ~ 05-12"). 시작과 끝 연도가 다르면 양쪽에 모두 적는다.
const range = (u: UserInfo) => {
  const a = fmtDate(tsToMs(u.start)).replace(/\./g, "-");
  const b = fmtDate(tsToMs(u.end)).replace(/\./g, "-");
  return a.slice(0, 4) === b.slice(0, 4) ? `${a} ~ ${b.slice(5)}` : `${a} ~ ${b}`;
};
const PSEUDONYM_NOTE = "개인정보 보호를 위해 원본 ID 대신 가명(U01~, S001~)을 씁니다.";
// 수면 기록 여부는 Stanford 참가자만 알려 준다(FitBit 목록은 모두 수면 기록이 있어 표시하지 않는다).
const sleepLabel = (u: UserInfo) => (u.dataset === "stanford" ? (u.has_sleep === false ? "수면 기록 없음" : "수면 기록 있음") : null);
const who = (u: UserInfo) => {
  const n = participantNo(u.user_id);
  return n == null ? null : `참가자 ${n}`;
};

/** 사용자 선택 드롭다운. */
export function UserSwitch({ users, value, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const [active, setActiveState] = useState(0);
  // 키 반복·연속 입력이 렌더보다 빨라도 항상 최신 활성 위치에서 이동하도록 ref 로 미러링한다.
  const activeRef = useRef(0);
  const setActive = (i: number) => {
    activeRef.current = i;
    setActiveState(i);
  };
  const root = useRef<HTMLDivElement>(null);
  const listId = useId();
  const selIdx = Math.max(0, users.findIndex((u) => u.user_id === value));
  const cur = users[selIdx];

  // 바깥을 누르면 닫는다.
  useEffect(() => {
    if (!open) return;
    const h = (e: PointerEvent) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", h);
    return () => document.removeEventListener("pointerdown", h);
  }, [open]);

  const openList = () => {
    setActive(selIdx);
    setOpen(true);
  };
  const pick = (i: number) => {
    onChange(users[i].user_id);
    setOpen(false);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (!users.length) return;
    if (!open) {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(e.key)) {
        e.preventDefault();
        openList();
      }
      return;
    }
    const move = (to: number) => {
      e.preventDefault();
      setActive(Math.max(0, Math.min(users.length - 1, to)));
    };
    const at = activeRef.current;
    if (e.key === "ArrowDown") move(at + 1);
    else if (e.key === "ArrowUp") move(at - 1);
    else if (e.key === "Home") move(0);
    else if (e.key === "End") move(users.length - 1);
    else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      pick(at);
    } else if (e.key === "Escape") {
      // 전역 Esc 단축키(확대 해제)로 번지지 않도록 여기서 소비한다.
      e.preventDefault();
      e.stopPropagation();
      setOpen(false);
    } else if (e.key === "Tab") setOpen(false);
  };

  return (
    <div className="picker" ref={root}>
      <span className="picker-label">대상</span>
      <div className="picker-field">
      <button
        type="button"
        role="combobox"
        className="picker-btn"
        aria-label="대상 사용자 선택"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listId}
        aria-activedescendant={open ? `${listId}-${active}` : undefined}
        disabled={!users.length}
        title={PSEUDONYM_NOTE}
        data-testid="user-select"
        onClick={() => (open ? setOpen(false) : openList())}
        onKeyDown={onKeyDown}
      >
        {cur ? (
          <>
            <span className="mono" translate="no">{cur.user_id}</span>
            {who(cur) && <span className="picker-who">{who(cur)}</span>}
            <span className="picker-range">{range(cur)}</span>
            {sleepLabel(cur) && <span className="picker-who">{sleepLabel(cur)}</span>}
          </>
        ) : (
          <span className="muted">불러오는 중…</span>
        )}
        <ChevronDown size={16} aria-hidden="true" className="picker-chev" />
      </button>
      {open && (
        <ul className="picker-menu" id={listId} role="listbox" aria-label="사용자 목록">
          {users.map((u, i) => (
            <li
              key={u.user_id}
              id={`${listId}-${i}`}
              role="option"
              aria-selected={u.user_id === value}
              className={`picker-opt ${i === active ? "active" : ""}`}
              data-testid="user-option"
              data-user-id={u.user_id}
              onPointerEnter={() => setActive(i)}
              // pointerdown 에서 선택한다: click 은 바깥 클릭 감지(pointerdown)와 겹쳐 목록이 먼저 닫힐 수 있다.
              onPointerDown={(e) => {
                e.preventDefault();
                pick(i);
              }}
            >
              <span className="mono" translate="no">{u.user_id}</span>
              <span>{who(u)}</span>
              <span className="mono">{range(u)}</span>
              <span className="mono r">{u.days}일</span>
              <span className="mono r">기록 {u.coverage_pct.toFixed(0)}%</span>
              {sleepLabel(u) && <span className="r muted">{sleepLabel(u)}</span>}
            </li>
          ))}
          <li role="presentation" className="picker-note">{PSEUDONYM_NOTE}</li>
        </ul>
      )}
      </div>
    </div>
  );
}
