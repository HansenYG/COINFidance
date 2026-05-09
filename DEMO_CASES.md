# COINFidance — Demo Cases

A curated set of **real, publicly documented** addresses you can paste into the
Wallet Analyzer and Coin Checker to showcase the product. Every entry below is
linked to a public source (Etherscan label, BscScan label, news article, or
on-chain forensic report).

> ⚠️  These are real addresses on public blockchains. **Never** send funds to
> the malicious ones — they are listed solely for analytical demos. Always
> double-check the address you copy.

---

## 🟢 Wallet Analyzer — Legitimate / Clean Wallets

Use these to demonstrate that the model returns "Normal" with low suspicion
scores. They have long histories, many unique counterparties, low fail
ratios, and varied inter-transaction time deltas.

| # | Wallet                                                                         | Label / Notes |
|---|--------------------------------------------------------------------------------|---------------|
| 1 | `0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045`                                   | **vitalik.eth** — Vitalik Buterin's primary public wallet. Years of activity, hundreds of unique receivers, low fail ratio. |
| 2 | `0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8`                                   | **Binance 7** — major exchange hot wallet. High volume, broad distribution. |
| 3 | `0x28C6c06298d514Db089934071355E5743bf21d60`                                   | **Binance 14** — another exchange wallet, different traffic pattern. |
| 4 | `0xF977814e90dA44bFA03b6295A0616a897441aceC`                                   | **Binance 8** — large hot wallet, high tx count. |

## 🔴 Wallet Analyzer — Malicious / Phishing / Hacker Wallets

Use these to demonstrate the model flagging "Suspicious" with high confidence.
All are tagged as Phish/Hack on Etherscan or linked to documented exploits.

| # | Wallet                                                                         | Label / Notes |
|---|--------------------------------------------------------------------------------|---------------|
| 1 | `0x98b0811E2CC7530380cAf1A17440B18F71f51f4e`                                   | **Fake_Phishing532118** (zachxbt-tagged). Social-engineering attack, ~$17.9 M ETH. |
| 2 | `0x00000c07575Bb4e64457687A0382b4D3Ea470000`                                   | **Fake_Phishing184810** — vanity-prefix phishing wallet (reported by Scam Sniffer). |
| 3 | `0x92d25fdb0e301225984CcA0c659Cfd5f5C07707A`                                   | **Fake_Phishing458824** — address-poisoning scammer. |
| 4 | `0x7188F241F230B577FAAbD4A2932d9B106F4504f1`                                   | **Fake_Phishing2512024** — bursty 0-value address-poisoning transfers (clear bot pattern). |
| 5 | `0x339f88D3e5c93fa83DBbC5911eaf3eAB28F0cb9b`                                   | **Fake_Phishing2042025** — address-poisoning, high failed-tx ratio. |
| 6 | `0x6d9052b2DF589De00324127fe2707eb34e592e48`                                   | The address that received a victim's mistakenly copied **4,556 ETH ($12.25 M)** address-poisoning transfer (reported by Scam Sniffer, 2025). |
| 7 | `0xe2ca471124b124831e231fb835778840ad100f97`                                   | **Gala Games Exploiter** — the actor who minted 5 B GALA tokens in May 2024 (~$200 M nominal). |

### 🧪 Suggested demo script — Wallet Analyzer

1. Paste **vitalik.eth** (`0xd8dA6BF2...96045`) → expect *Normal*, low score.
2. Paste **Fake_Phishing2512024** (`0x7188F241...04f1`) → expect *Suspicious*,
   high score (it has the bursty 0-value pattern the model is trained on).
3. Show how the dashboard "Recent Wallet Scans" updates with both results,
   one badge green ("Normal") and one red ("Suspicious"), pulled from
   Supabase.

---

## 🟢 Coin Checker — Legitimate Tokens

Long-lived, audited, deeply liquid tokens. Should classify as *Legitimate*.

| # | Token       | Contract address (Ethereum)                                              | Notes |
|---|-------------|--------------------------------------------------------------------------|-------|
| 1 | **USDC**    | `0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48`                             | Circle USD stablecoin. |
| 2 | **USDT**    | `0xdAC17F958D2ee523a2206206994597C13D831ec7`                             | Tether USD. |
| 3 | **WETH**    | `0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2`                             | Wrapped Ether. |
| 4 | **LINK**    | `0x514910771AF9Ca656af840dff83E8264EcF986CA`                             | Chainlink. |
| 5 | **UNI**     | `0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984`                             | Uniswap governance token. |
| 6 | **GALA v2** | `0xd1d2Eb1B1e90B638588728b4130137D262C87cae`                             | Gala Games official token (proxy contract, multi-sig protected). |

## 🔴 Coin Checker — Known Scam / Rug-pull Tokens

Either confirmed scams, rug-pulled tokens, or contracts with malicious
"honey-pot" code preventing users from selling.

| # | Token            | Contract address                                                       | Notes |
|---|------------------|------------------------------------------------------------------------|-------|
| 1 | **SQUID** (Eth)  | `0x57D9750893ADB0d1ee07bac44c8bB45c45b58f73`                           | Squid Game token rug-pull (~$3.4 M, Nov 2021). Holders couldn't sell — the most cited honey-pot in crypto history. |
| 2 | **SQUID** (BSC)  | `0x1f5Eabba9c56BcA4a7828969B79bc87051125b31`  *(BscScan)*               | Same scam on BNB Chain — labelled "SQUID Token Rug 1" by BscScan. |
| 3 | **SQUIDGAME**    | `0x9cd67127b638b2074bc6523aefa3c04f7a729038`                           | 2024 copycat scam — contract source contains hidden `_dogeswap` external call to an attacker-controlled router (function `grok27goat38dent`) that effectively blocks transfers. Classic honey-pot. |
| 4 | **PEPE-clone scams** | Search Etherscan for tokens labelled "Phish / Hack" with name `PEPE` | Many tokens impersonate PEPE — useful for showing brand-impersonation detection. |

> Tip — to surface more scam tokens on demand, search Etherscan's
> [Label Cloud → "Heist"](https://etherscan.io/labelcloud) or
> [scam-token aggregator GoPlus](https://gopluslabs.io/token-security/).

### 🧪 Suggested demo script — Coin Checker

1. Paste **USDC** (`0xA0b86991...eB48`) → expect *Legitimate*, low risk score.
2. Paste **SQUID** (`0x57D97508...58f73`) → expect *Scam* with high confidence.
3. Paste **SQUIDGAME** (`0x9cd67127...729038`) → demonstrate detection of a
   modern honey-pot impersonator.
4. Open the Supabase-backed "Coin Scans" history on the dashboard to show
   the persisted results.

---

## 📰 Background reading (for talking points during the demo)

| Incident | Date | Loss | Reference |
|----------|------|------|-----------|
| **Squid Game token** rug-pull | Nov 2021 | ~$3.4 M (BNB), $19.3 M total across linked rugs | [TRM Labs report](https://www.trmlabs.com/post/on-the-trail-of-the-squid-game-scammers) |
| **Gala Games** unauthorized mint | May 2024 | $21.8 M cashed out (5 B GALA minted) | [Gala official report](https://news.gala.com/galachain/incident-report-unauthorized-token-minting-blockchain-game-partners-inc/) |
| **$12.25 M ETH address-poisoning** | 2025 | $12.25 M ETH | [Scam Sniffer / BTCC writeup](https://www.btcc.com/en-US/square/LedgerSpectre/1472120) |
| **$50 M USDT address-poisoning** | Dec 2025 | $49.99 M USDT | Scam Sniffer (2025) |
| **Etherscan ad phishing campaign** | Apr 2024 | ~$300 M (industry-wide, 2023) | [Cointelegraph](https://cointelegraph.com/news/etherscan-ads-phishing-campaign) |

---

## 🛠 Quick copy-paste block (most useful demos)

```
# Clean wallet
0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045    # vitalik.eth

# Suspicious wallet (bot-like address-poisoning pattern)
0x7188F241F230B577FAAbD4A2932d9B106F4504f1    # Fake_Phishing2512024

# Legit token
0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48    # USDC

# Scam token
0x57D9750893ADB0d1ee07bac44c8bB45c45b58f73    # SQUID rug-pull
```
